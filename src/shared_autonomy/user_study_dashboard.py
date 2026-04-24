#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web UI for task-oriented user study flow."""

import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
import yaml
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>User Study Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #172033;
      --line: #263449;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #60a5fa;
      --accent-2: #38bdf8;
      --ok: #22c55e;
      --warn: #f59e0b;
      --danger: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top, #1d4ed8 0%, #0f172a 35%);
    }
    .page {
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px;
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
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .nav-tabs {
      display: inline-flex;
      gap: 8px;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(17, 24, 39, 0.94);
      box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    }
    .nav-tab {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      color: var(--muted);
      background: transparent;
    }
    .nav-tab.active {
      color: white;
      background: linear-gradient(135deg, var(--accent), #2563eb);
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .card {
      background: rgba(17, 24, 39, 0.94);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    }
    .metric-label {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .metric-value {
      font-size: 32px;
      font-weight: 700;
      margin-top: 6px;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
    }
    .section-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .section-subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .task-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .task-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(23, 32, 51, 0.78);
      cursor: pointer;
      transition: border-color .15s ease, transform .15s ease;
    }
    .task-card:hover { border-color: var(--accent-2); transform: translateY(-1px); }
    .task-card.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px rgba(96,165,250,.4); }
    .task-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .task-desc {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      min-height: 56px;
    }
    .task-actions {
      margin-top: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), #2563eb);
    }
    button.secondary { background: #334155; }
    button.warn { background: linear-gradient(135deg, #f59e0b, #d97706); }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .pipeline {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .step {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.76);
    }
    .step.active { border-color: var(--accent); }
    .step.done { border-color: rgba(34,197,94,.55); background: rgba(20,42,34,.76); }
    .step.manual { border-style: dashed; }
    .step-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
    }
    .step-name { font-weight: 700; }
    .badge {
      font-size: 12px;
      font-weight: 700;
      border-radius: 999px;
      padding: 4px 8px;
    }
    .badge.active { background: rgba(96,165,250,.18); color: #bfdbfe; }
    .badge.done { background: rgba(34,197,94,.16); color: #86efac; }
    .badge.pending { background: rgba(148,163,184,.18); color: #cbd5e1; }
    .step-desc {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .prompt-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(13,24,42,.94) 0%, rgba(18,31,56,.94) 100%);
      margin-bottom: 12px;
      min-height: 116px;
    }
    .prompt-title {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .06em;
      font-size: 12px;
      margin-bottom: 8px;
    }
    .prompt-text {
      font-size: 18px;
      line-height: 1.5;
    }
    .status-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .status-item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.78);
    }
    .status-k {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .status-v {
      font-size: 18px;
      font-weight: 700;
      margin-top: 6px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .control-chip {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.78);
    }
    .control-chip strong {
      display: block;
      margin-bottom: 6px;
      color: #bfdbfe;
    }
    .control-chip span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .notice {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .view {
      display: none;
    }
    .view.active {
      display: block;
    }
    .instruction-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }
    .instruction-list {
      display: grid;
      gap: 10px;
    }
    .instruction-item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.78);
    }
    .instruction-item strong {
      display: block;
      margin-bottom: 6px;
      color: #bfdbfe;
    }
    .instruction-item span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .joystick-wrap {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 540px;
      padding: 8px 0;
    }
    .joystick-svg {
      width: 100%;
      max-width: 560px;
      height: auto;
    }
    .joystick-shell {
      fill: #0b1220;
      stroke: #314156;
      stroke-width: 3;
    }
    .joystick-panel {
      fill: #121b2d;
      stroke: #314156;
      stroke-width: 2;
    }
    .stick-base {
      fill: #1f2937;
      stroke: #475569;
      stroke-width: 2;
    }
    .stick-cap {
      fill: #334155;
      stroke: #64748b;
      stroke-width: 2;
    }
    .face-a { fill: #22c55e; }
    .face-b { fill: #ef4444; }
    .face-x { fill: #3b82f6; }
    .face-y { fill: #f59e0b; }
    .joystick-line {
      stroke: #60a5fa;
      stroke-width: 3;
      fill: none;
      stroke-linecap: round;
    }
    .joystick-label {
      fill: #e5e7eb;
      font-size: 13px;
      font-weight: 700;
    }
    .joystick-note {
      fill: #94a3b8;
      font-size: 11px;
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 1fr; }
      .instruction-layout { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .summary, .task-grid, .status-list, .controls { grid-template-columns: 1fr; }
      .topbar { align-items: stretch; }
      .nav-tabs { width: 100%; }
      .nav-tab { flex: 1; }
      .joystick-wrap { min-height: 0; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="title">User Study Dashboard</div>
    <div class="subtitle">Task-oriented guidance for shared-autonomy grasp selection and execution.</div>

    <div class="topbar">
      <div class="nav-tabs" role="tablist" aria-label="Dashboard Views">
        <button class="nav-tab active" id="tab-dashboard" data-view="dashboard" type="button">Live Dashboard</button>
        <button class="nav-tab" id="tab-instructions" data-view="instructions" type="button">Instructions</button>
      </div>
      <div class="notice">Use the instructions view as a quick operator reference before starting a task.</div>
    </div>

    <div class="view active" id="view-dashboard">
      <div class="summary">
        <div class="card">
          <div class="metric-label">Active Task</div>
          <div class="metric-value" id="active-task">None</div>
        </div>
        <div class="card">
          <div class="metric-label">Current Phase</div>
          <div class="metric-value" id="current-phase">scan_workspace</div>
        </div>
        <div class="card">
          <div class="metric-label">Top Candidate</div>
          <div class="metric-value" id="top-goal">none</div>
        </div>
        <div class="card">
          <div class="metric-label">Confidence</div>
          <div class="metric-value" id="top-prob">0%</div>
        </div>
      </div>

      <div class="layout">
        <div>
          <div class="card" style="margin-bottom: 18px;">
            <div class="section-title">Choose Task</div>
            <div class="section-subtitle">Selecting a task starts a guided phase pipeline and constrains intent recognition to the relevant object group.</div>
            <div class="task-grid" id="task-grid"></div>
            <div class="task-actions">
              <button class="secondary" id="reset-btn">Reset To Scan</button>
              <button class="warn" id="manual-btn" disabled>Complete Manual Step</button>
            </div>
          </div>

          <div class="card">
            <div class="section-title">Task Pipeline</div>
            <div class="section-subtitle">The interface automatically advances to the next automated step after a grasp completes.</div>
            <div class="pipeline" id="pipeline"></div>
          </div>
        </div>

        <div>
          <div class="card" style="margin-bottom: 18px;">
            <div class="section-title">Current Guidance</div>
            <div class="section-subtitle">These prompts are driven by the current phase, intent inference, and execution state.</div>
            <div class="prompt-box">
              <div class="prompt-title">Task Prompt</div>
              <div class="prompt-text" id="task-prompt">Waiting for task selection.</div>
            </div>
            <div class="prompt-box">
              <div class="prompt-title">Execution Prompt</div>
              <div class="prompt-text" id="exec-prompt">Move the wrist camera to discover tags.</div>
            </div>
            <div class="status-list">
              <div class="status-item">
                <div class="status-k">Selected Label</div>
                <div class="status-v" id="selected-label">none</div>
              </div>
              <div class="status-item">
                <div class="status-k">Execution State</div>
                <div class="status-v" id="execution-state">idle</div>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="section-title">Joystick Controls</div>
            <div class="section-subtitle">Use the joystick to indicate intent and execute each step.</div>
            <div class="controls">
              <div class="control-chip">
                <strong>Left Stick</strong>
                <span>Move the arm in the horizontal plane to express intent toward the target object.</span>
              </div>
              <div class="control-chip">
                <strong>Right Stick Vertical</strong>
                <span>Adjust end-effector height while exploring the workspace.</span>
              </div>
              <div class="control-chip">
                <strong>X Button</strong>
                <span>Confirm and execute the next stage when prompted.</span>
              </div>
              <div class="control-chip">
                <strong>Y Button</strong>
                <span>Cancel the current execution and return to the selection state.</span>
              </div>
              <div class="control-chip">
                <strong>A Button</strong>
                <span>Close the gripper when direct teleoperation requires it.</span>
              </div>
              <div class="control-chip">
                <strong>B Button</strong>
                <span>Open the gripper when direct teleoperation requires it.</span>
              </div>
            </div>
            <div class="notice" id="task-note" style="margin-top: 12px;"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="view" id="view-instructions">
      <div class="instruction-layout">
        <div class="card">
          <div class="section-title">Operator Instructions</div>
          <div class="section-subtitle">This page is a quick reference for the current AprilTag user-study flow.</div>
          <div class="instruction-list">
            <div class="instruction-item">
              <strong>1. Start in scan mode</strong>
              <span>Reset to scan, then move the wrist camera until the relevant AprilTags are visible and stable.</span>
            </div>
            <div class="instruction-item">
              <strong>2. Pick a task</strong>
              <span>Choose the task on the live dashboard. The pipeline will restrict intent inference to the valid object set for that step.</span>
            </div>
            <div class="instruction-item">
              <strong>3. Express intent with the arm</strong>
              <span>Use joystick motion to bias the end-effector toward the intended object. Watch the top candidate and confidence before confirming.</span>
            </div>
            <div class="instruction-item">
              <strong>4. Confirm only when prompted</strong>
              <span>Press X when the execution prompt indicates the system is ready for the next stage. Use Y to cancel if the target or pose is wrong.</span>
            </div>
            <div class="instruction-item">
              <strong>5. Manual steps stay manual</strong>
              <span>For guided non-automatic steps, complete the physical action first and then use the manual-step button on the live dashboard.</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="section-title">Labeled Joystick</div>
          <div class="section-subtitle">Current button mappings from the active shared-autonomy launch.</div>
          <div class="joystick-wrap">
            <svg class="joystick-svg" viewBox="0 0 560 520" role="img" aria-label="Xbox-style joystick with labeled controls">
              <path class="joystick-shell" d="M150 126c26-24 57-36 96-36h68c39 0 70 12 96 36l36 34c24 23 34 58 28 91l-18 93c-5 26-24 46-49 53-23 6-47-1-63-19l-43-47H259l-43 47c-16 18-40 25-63 19-25-7-44-27-49-53l-18-93c-6-33 4-68 28-91z"/>
              <rect class="joystick-panel" x="154" y="152" width="252" height="154" rx="26"/>
              <circle class="stick-base" cx="206" cy="214" r="46"/>
              <circle class="stick-cap" cx="206" cy="214" r="28"/>
              <circle class="stick-base" cx="314" cy="258" r="46"/>
              <circle class="stick-cap" cx="314" cy="258" r="28"/>
              <rect class="stick-base" x="236" y="194" width="18" height="58" rx="8"/>
              <rect class="stick-base" x="216" y="214" width="58" height="18" rx="8"/>
              <circle class="face-y" cx="420" cy="178" r="18"/>
              <circle class="face-b" cx="456" cy="214" r="18"/>
              <circle class="face-a" cx="420" cy="250" r="18"/>
              <circle class="face-x" cx="384" cy="214" r="18"/>
              <path class="joystick-line" d="M420 178 L502 124 L534 124"/>
              <text class="joystick-label" x="540" y="120" text-anchor="end">Y: Cancel current execution</text>
              <text class="joystick-note" x="540" y="138" text-anchor="end">Return to target selection</text>
              <path class="joystick-line" d="M384 214 L56 126 L24 126"/>
              <text class="joystick-label" x="20" y="122" text-anchor="start">X: Confirm next stage</text>
              <text class="joystick-note" x="20" y="140" text-anchor="start">Use when prompted</text>
              <path class="joystick-line" d="M420 250 L502 314 L534 314"/>
              <text class="joystick-label" x="540" y="310" text-anchor="end">A: Close gripper</text>
              <text class="joystick-note" x="540" y="328" text-anchor="end">Direct teleop only</text>
              <path class="joystick-line" d="M456 214 L504 214 L534 214"/>
              <text class="joystick-label" x="540" y="210" text-anchor="end">B: Open gripper</text>
              <text class="joystick-note" x="540" y="228" text-anchor="end">Direct teleop only</text>
              <path class="joystick-line" d="M206 214 L86 214 L24 214"/>
              <text class="joystick-label" x="20" y="210" text-anchor="start">Left Stick: Intent motion</text>
              <text class="joystick-note" x="20" y="228" text-anchor="start">Move in-plane toward target</text>
              <path class="joystick-line" d="M314 258 L210 404 L24 404"/>
              <text class="joystick-label" x="20" y="400" text-anchor="start">Right Stick Vertical: Height</text>
              <text class="joystick-note" x="20" y="418" text-anchor="start">Raise or lower end-effector</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = { data: null };

    function probPct(v) {
      return Number.isFinite(v) ? `${Math.round(v * 100)}%` : "0%";
    }

    function setView(view) {
      const isInstructions = view === "instructions";
      document.getElementById("view-dashboard").classList.toggle("active", !isInstructions);
      document.getElementById("view-instructions").classList.toggle("active", isInstructions);
      document.getElementById("tab-dashboard").classList.toggle("active", !isInstructions);
      document.getElementById("tab-instructions").classList.toggle("active", isInstructions);
      window.location.hash = isInstructions ? "#instructions" : "#dashboard";
    }

    async function postJSON(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {})
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      return res.json();
    }

    function renderTasks(data) {
      const root = document.getElementById("task-grid");
      const tasks = Object.values(data.tasks || {});
      root.innerHTML = tasks.map((task) => {
        const active = data.active_task && data.active_task.id === task.id ? "active" : "";
        return `
          <div class="task-card ${active}" data-task="${task.id}">
            <div class="task-title">${task.display_name}</div>
            <div class="task-desc">${task.description || ""}</div>
          </div>
        `;
      }).join("");
      root.querySelectorAll(".task-card").forEach((el) => {
        el.addEventListener("click", async () => {
          await postJSON("/api/start_task", {task_id: el.dataset.task});
          await refreshState();
        });
      });
    }

    function renderPipeline(data) {
      const root = document.getElementById("pipeline");
      const activeTask = data.active_task;
      if (!activeTask) {
        root.innerHTML = '<div class="step"><div class="step-desc">No task selected. Choose a task to begin the guided pipeline.</div></div>';
        document.getElementById("manual-btn").disabled = true;
        return;
      }

      root.innerHTML = activeTask.steps.map((step, idx) => {
        const statusClass = step.status === "done" ? "done" : (step.status === "active" ? "active" : "");
        const manualClass = step.manual ? "manual" : "";
        return `
          <div class="step ${statusClass} ${manualClass}">
            <div class="step-head">
              <div class="step-name">${idx + 1}. ${step.title}</div>
              <div class="badge ${step.status}">${step.status}</div>
            </div>
            <div class="step-desc">${step.description || ""}</div>
          </div>
        `;
      }).join("");

      const activeStep = activeTask.steps.find((step) => step.status === "active");
      document.getElementById("manual-btn").disabled = !(activeStep && activeStep.manual);
    }

    function render(data) {
      document.getElementById("active-task").textContent = data.active_task ? data.active_task.display_name : "None";
      document.getElementById("current-phase").textContent = data.current_phase || "scan_workspace";
      document.getElementById("top-goal").textContent = data.top_goal_label || "none";
      document.getElementById("top-prob").textContent = probPct(data.top_probability);
      document.getElementById("task-prompt").textContent = data.task_prompt || "Waiting for task selection.";
      document.getElementById("exec-prompt").textContent = data.execution_prompt || "Move the joystick to indicate intent.";
      document.getElementById("selected-label").textContent = data.selected_grasp_label || "none";
      document.getElementById("execution-state").textContent = data.execution_state || "idle";
      document.getElementById("task-note").textContent = data.active_task ? (data.active_task.note || "") : "";

      renderTasks(data);
      renderPipeline(data);
    }

    async function refreshState() {
      try {
        const res = await fetch("/api/state");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.data = await res.json();
        render(state.data);
      } catch (err) {
        console.error("dashboard refresh failed", err);
      }
    }

    document.getElementById("reset-btn").addEventListener("click", async () => {
      await postJSON("/api/reset_task", {});
      await refreshState();
    });

    document.getElementById("manual-btn").addEventListener("click", async () => {
      await postJSON("/api/manual_advance", {});
      await refreshState();
    });

    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    });

    window.addEventListener("hashchange", () => {
      setView(window.location.hash === "#instructions" ? "instructions" : "dashboard");
    });

    setView(window.location.hash === "#instructions" ? "instructions" : "dashboard");
    refreshState();
    setInterval(refreshState, 700);
  </script>
</body>
</html>
"""


class UserStudyDashboard:
    def __init__(self):
        rospy.init_node("user_study_dashboard")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.tasks_yaml = os.path.expanduser(
            rospy.get_param("~tasks_yaml", os.path.join(package_root, "config", "user_study_tasks.yaml"))
        )
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", os.path.join(package_root, "config", "apriltag_object_map.yaml"))
        )
        self.host = str(rospy.get_param("~host", "127.0.0.1")).strip()
        self.port = int(rospy.get_param("~port", 8766))
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.task_prompt_topic = str(rospy.get_param("~task_prompt_topic", "/task_context/prompt")).strip()
        self.execution_prompt_topic = str(rospy.get_param("~execution_prompt_topic", "/apriltag_executor/prompt")).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_prob_topic = str(rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")).strip()
        self.distribution_topic = str(rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")).strip()
        self.candidates_topic = str(rospy.get_param("~candidates_topic", "/apriltag_grasp_registry/detections")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.initial_reset_command = str(rospy.get_param("~initial_reset_command", "scan_workspace")).strip()
        default_log_dir = os.path.join(package_root, "logs")
        self.probability_log_dir = os.path.expanduser(
            rospy.get_param("~probability_log_dir", default_log_dir)
        )

        self.lock = threading.Lock()
        self.tasks = self._load_tasks()
        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()
        self.active_task_id = None
        self.active_step_index = None
        self.current_phase = self.initial_reset_command
        self.task_prompt = "Waiting for task selection."
        self.execution_prompt = "Move the joystick to indicate intent."
        self.execution_state = "idle"
        self.top_goal_label = ""
        self.top_probability = 0.0
        self.selected_grasp_label = ""
        self.allowed_tag_ids = set()
        self.latest_candidate_labels = []
        self.last_distribution = []
        self.last_distribution_stamp = ""
        self.probability_log_path = self._make_probability_log_path()

        self.command_pub = rospy.Publisher(self.command_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=1)
        rospy.Subscriber(self.task_prompt_topic, String, self._task_prompt_cb, queue_size=1)
        rospy.Subscriber(self.execution_prompt_topic, String, self._execution_prompt_cb, queue_size=1)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)
        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=1)
        rospy.Subscriber(self.top_prob_topic, Float32, self._top_prob_cb, queue_size=1)
        rospy.Subscriber(self.distribution_topic, Float32MultiArray, self._distribution_cb, queue_size=10)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_label_cb, queue_size=1)

        self.httpd = self._make_server()
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        rospy.loginfo("[user_study_dashboard] ready at http://%s:%d", self.host, self.port)

    def _load_tasks(self):
        if not os.path.exists(self.tasks_yaml):
            raise RuntimeError("Tasks YAML not found: {}".format(self.tasks_yaml))
        with open(self.tasks_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        tasks = {}
        for task_id, meta in (raw.get("tasks", {}) or {}).items():
            task = dict(meta or {})
            task["id"] = str(task_id)
            task["steps"] = list(task.get("steps", []))
            tasks[task["id"]] = task
        return tasks

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}, {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        label_map = {}
        tag_id_map = {}
        for key, meta in (raw.get("tag_objects", {}) or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                tag_id = int(key)
            except Exception:
                tag_id = None
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                label_map[label] = meta
            if tag_id is not None:
                tag_id_map[str(tag_id)] = meta
        return label_map, tag_id_map

    def _make_probability_log_path(self):
        os.makedirs(self.probability_log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.probability_log_dir, "user_study_probability_{}.jsonl".format(stamp))

    def _phase_cb(self, msg):
        with self.lock:
            self.current_phase = str(msg.data).strip()

    def _task_prompt_cb(self, msg):
        with self.lock:
            self.task_prompt = str(msg.data).strip()

    def _execution_prompt_cb(self, msg):
        with self.lock:
            self.execution_prompt = str(msg.data).strip()

    def _top_goal_cb(self, msg):
        with self.lock:
            self.top_goal_label = str(msg.data).strip()

    def _top_prob_cb(self, msg):
        with self.lock:
            self.top_probability = float(msg.data)

    def _selected_label_cb(self, msg):
        with self.lock:
            self.selected_grasp_label = str(msg.data).strip()

    def _allowed_ids_cb(self, msg):
        with self.lock:
            self.allowed_tag_ids = set(int(v) for v in list(msg.data))

    def _candidates_cb(self, msg):
        labels = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            labels.append(str(int(hyp.id)))
        with self.lock:
            self.latest_candidate_labels = labels

    def _current_step_locked(self):
        task = self._current_task()
        if task is None or self.active_step_index is None:
            return None
        if self.active_step_index < 0 or self.active_step_index >= len(task["steps"]):
            return None
        return task["steps"][self.active_step_index]

    def _distribution_labels_locked(self, count):
        labels = list(self.latest_candidate_labels)
        if self.allowed_tag_ids:
            filtered = []
            for label in labels:
                try:
                    if int(label) in self.allowed_tag_ids:
                        filtered.append(label)
                except Exception:
                    continue
            labels = filtered
        return labels[:count]

    def _tag_meta_for_label(self, label):
        meta = self.tag_id_to_meta.get(str(label), {})
        if not isinstance(meta, dict):
            return {}
        return meta

    def _append_probability_log(self, entry):
        with open(self.probability_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _distribution_cb(self, msg):
        with self.lock:
            probs = [float(v) for v in list(msg.data)]
            labels = self._distribution_labels_locked(len(probs))
            self.last_distribution = probs
            self.last_distribution_stamp = datetime.now().isoformat()
            task = self._current_task()
            step = self._current_step_locked()
            entry = {
                "timestamp": self.last_distribution_stamp,
                "active_task_id": None if task is None else task["id"],
                "active_task_name": None if task is None else str(task.get("display_name", task["id"])),
                "active_step_index": self.active_step_index,
                "active_step_id": None if step is None else str(step.get("id", "")),
                "active_step_title": None if step is None else str(step.get("title", "")),
                "current_phase": self.current_phase,
                "allowed_tag_ids": sorted(self.allowed_tag_ids),
                "candidate_labels": labels,
                "candidate_objects": [
                    {
                        "label": label,
                        "object_name": str(self._tag_meta_for_label(label).get("object_name", "")),
                        "category": str(self._tag_meta_for_label(label).get("category", "")),
                    }
                    for label in labels
                ],
                "probabilities": probs[:len(labels)],
                "top_goal_label": self.top_goal_label,
                "top_goal_object_name": str(self._tag_meta_for_label(self.top_goal_label).get("object_name", "")),
                "top_probability": self.top_probability,
                "selected_grasp_label": self.selected_grasp_label,
                "execution_state": self.execution_state,
            }
        try:
            self._append_probability_log(entry)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[user_study_dashboard] failed to append probability log: %s", exc)

    def _execution_state_cb(self, msg):
        text = str(msg.data).strip()
        with self.lock:
            self.execution_state = text
        self._handle_execution_transition(text)

    def _publish_command(self, cmd):
        if not cmd:
            return
        self.command_pub.publish(String(data=str(cmd)))

    def _current_task(self):
        if self.active_task_id is None:
            return None
        return self.tasks.get(self.active_task_id)

    def _activate_step_locked(self, step_index):
        task = self._current_task()
        if task is None:
            return
        self.active_step_index = step_index
        self.selected_grasp_label = ""
        self.execution_state = "idle"
        step = task["steps"][step_index]
        command = str(step.get("command", "")).strip()
        if command:
            self._publish_command(command)

    def _reset_locked(self):
        self.active_task_id = None
        self.active_step_index = None
        self.selected_grasp_label = ""
        self.execution_state = "idle"
        self._publish_command(self.initial_reset_command)

    def start_task(self, task_id):
        with self.lock:
            if task_id not in self.tasks:
                raise KeyError(task_id)
            self.active_task_id = task_id
            self.active_step_index = None
            task = self.tasks[task_id]
            if task["steps"]:
                self._activate_step_locked(0)

    def reset_task(self):
        with self.lock:
            self._reset_locked()

    def manual_advance(self):
        with self.lock:
            task = self._current_task()
            if task is None or self.active_step_index is None:
                return
            step = task["steps"][self.active_step_index]
            if not bool(step.get("manual", False)):
                return
            self._advance_locked()

    def _advance_locked(self):
        task = self._current_task()
        if task is None or self.active_step_index is None:
            return
        next_index = self.active_step_index + 1
        if next_index >= len(task["steps"]):
            self.active_step_index = None
            self.active_task_id = None
            self._publish_command(str(task.get("completion_reset_command", self.initial_reset_command)).strip())
            return
        self._activate_step_locked(next_index)

    def _step_matches(self, step, execution_text):
        if not execution_text.startswith("grasp_complete:"):
            return False
        label = execution_text.split(":", 1)[1].strip()
        meta = self.label_to_meta.get(label, {})
        completion_label = str(step.get("completion_label", "")).strip()
        completion_category = str(step.get("completion_category", "")).strip()
        completion_categories = [
            str(item).strip()
            for item in list(step.get("completion_categories", []) or [])
            if str(item).strip()
        ]
        if completion_label and completion_label == label:
            return True
        if completion_category and str(meta.get("category", "")).strip() == completion_category:
            return True
        if completion_categories and str(meta.get("category", "")).strip() in completion_categories:
            return True
        return False

    def _handle_execution_transition(self, text):
        with self.lock:
            task = self._current_task()
            if task is None or self.active_step_index is None:
                return
            step = task["steps"][self.active_step_index]
            if self._step_matches(step, text):
                self._advance_locked()

    def _active_task_view_locked(self):
        task = self._current_task()
        if task is None:
            return None
        steps = []
        for index, step in enumerate(task["steps"]):
            status = "pending"
            if self.active_step_index is None:
                status = "done"
            elif index < self.active_step_index:
                status = "done"
            elif index == self.active_step_index:
                status = "active"
            steps.append(
                {
                    "id": str(step.get("id", "step_{}".format(index))),
                    "title": str(step.get("title", "Step {}".format(index + 1))),
                    "description": str(step.get("description", "")),
                    "manual": bool(step.get("manual", False)),
                    "status": status,
                }
            )
        return {
            "id": task["id"],
            "display_name": str(task.get("display_name", task["id"])),
            "description": str(task.get("description", "")),
            "note": "Current phase pipeline is driven by task_context commands and execution_state feedback.",
            "steps": steps,
        }

    def state_payload(self):
        with self.lock:
            task_views = {}
            for task_id, task in self.tasks.items():
                task_views[task_id] = {
                    "id": task_id,
                    "display_name": str(task.get("display_name", task_id)),
                    "description": str(task.get("description", "")),
                }
            return {
                "tasks": task_views,
                "active_task": self._active_task_view_locked(),
                "current_phase": self.current_phase,
                "task_prompt": self.task_prompt,
                "execution_prompt": self.execution_prompt,
                "execution_state": self.execution_state,
                "top_goal_label": self.top_goal_label,
                "top_probability": self.top_probability,
                "selected_grasp_label": self.selected_grasp_label,
                "probability_log_path": self.probability_log_path,
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

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}

                try:
                    if self.path == "/api/start_task":
                        dashboard.start_task(str(payload.get("task_id", "")).strip())
                    elif self.path == "/api/reset_task":
                        dashboard.reset_task()
                    elif self.path == "/api/manual_advance":
                        dashboard.manual_advance()
                    else:
                        self.send_error(404)
                        return
                except KeyError:
                    body = b'{"ok": false, "error": "unknown_task"}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                rospy.logdebug("[user_study_dashboard] " + fmt, *args)

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
    UserStudyDashboard().run()


if __name__ == "__main__":
    main()
