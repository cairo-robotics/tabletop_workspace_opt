#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web UI for task-oriented user study flow."""

import json
import os
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rospy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Twist
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray

try:
    from intera_interface import CHECK_VERSION, Limb, RobotEnable
except Exception:
    CHECK_VERSION = None
    Limb = None
    RobotEnable = None


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Participant Interface</title>
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
      display: none;
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
    .readiness-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .readiness-title {
      font-size: 18px;
      font-weight: 700;
    }
    .readiness-badge {
      font-size: 12px;
      font-weight: 700;
      border-radius: 999px;
      padding: 6px 10px;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .readiness-badge.ready { background: rgba(34,197,94,.16); color: #86efac; }
    .readiness-badge.needs_rescan { background: rgba(245,158,11,.18); color: #fcd34d; }
    .readiness-badge.idle { background: rgba(148,163,184,.18); color: #cbd5e1; }
    .readiness-message {
      color: var(--text);
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .readiness-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .readiness-stat {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.78);
    }
    .readiness-stat-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .readiness-stat-value {
      font-size: 22px;
      font-weight: 700;
      margin-top: 6px;
    }
    .readiness-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .readiness-chip {
      font-size: 12px;
      font-weight: 700;
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      background: rgba(15, 23, 42, 0.7);
    }
    .readiness-chip.recorded { border-color: rgba(34,197,94,.5); color: #86efac; }
    .readiness-chip.missing { border-color: rgba(245,158,11,.5); color: #fcd34d; }
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
    .participant-banner {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(17, 24, 39, 0.94);
      box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    }
    .operator-details {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(23, 32, 51, 0.78);
      overflow: hidden;
    }
    .operator-details summary {
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
      color: var(--muted);
      font-weight: 700;
    }
    .operator-details summary::-webkit-details-marker {
      display: none;
    }
    .operator-body {
      padding: 0 14px 14px 14px;
    }
    .participant-minimal .nav-tabs,
    .participant-minimal .summary,
    .participant-minimal #view-instructions,
    .participant-minimal #pipeline-card,
    .participant-minimal #task-status-card,
    .participant-minimal #scene-readiness-card,
    .participant-minimal #guidance-card,
    .participant-minimal .operator-details,
    .participant-minimal .task-focus {
      display: none !important;
    }
    .participant-minimal .layout {
      grid-template-columns: 1fr;
      max-width: 920px;
      margin: 0 auto;
    }
    .participant-minimal .page {
      max-width: 1120px;
    }
    .participant-minimal .task-grid {
      display: none !important;
    }
    .participant-minimal .page {
      max-width: 960px;
    }
    .participant-minimal .prompt-box {
      min-height: 0;
    }
    .participant-minimal .prompt-text {
      font-size: 30px;
      font-weight: 700;
      line-height: 1.22;
    }
    .participant-minimal .participant-banner {
      font-size: 14px;
      line-height: 1.5;
    }
    .participant-minimal .subtitle,
    .participant-minimal .participant-banner,
    .participant-minimal #manual-label-card,
    .participant-minimal #clear-sandwich-masks-btn {
      display: none !important;
    }
    .participant-question-card {
      margin-bottom: 18px;
      background: linear-gradient(180deg, rgba(13,24,42,.96) 0%, rgba(18,31,56,.96) 100%);
      border-width: 2px;
      transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
    }
    .participant-question-card.move {
      border-color: rgba(96,165,250,.7);
      box-shadow: 0 0 0 1px rgba(96,165,250,.18) inset;
    }
    .participant-question-card.locked {
      border-color: rgba(34,197,94,.8);
      box-shadow: 0 0 0 1px rgba(34,197,94,.18) inset;
      background: linear-gradient(180deg, rgba(13,36,30,.95) 0%, rgba(18,31,56,.95) 100%);
    }
    .participant-question-card.confirm {
      border-color: rgba(245,158,11,.85);
      box-shadow: 0 0 0 1px rgba(245,158,11,.18) inset;
      background: linear-gradient(180deg, rgba(55,33,10,.95) 0%, rgba(18,31,56,.95) 100%);
    }
    .participant-question-card.auto {
      border-color: rgba(34,197,94,.8);
      box-shadow: 0 0 0 1px rgba(34,197,94,.18) inset;
    }
    .participant-question-card.wait {
      border-color: rgba(148,163,184,.55);
    }
    .participant-cue {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 14px;
      font-size: 16px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 10px 18px;
      border: 1px solid var(--line);
      color: #cbd5e1;
      background: rgba(15, 23, 42, 0.78);
    }
    .participant-cue.move { color: #bfdbfe; border-color: rgba(96,165,250,.55); }
    .participant-cue.locked {
      color: #dcfce7;
      border-color: rgba(34,197,94,.90);
      background: linear-gradient(135deg, rgba(21,128,61,.38) 0%, rgba(22,101,52,.30) 45%, rgba(15,23,42,.88) 100%);
      box-shadow: 0 0 0 1px rgba(34,197,94,.20) inset, 0 0 26px rgba(34,197,94,.18);
    }
    .participant-cue.locked::before {
      content: "";
      width: 15px;
      height: 15px;
      border-radius: 999px;
      background: radial-gradient(circle at 35% 35%, #f0fdf4 0%, #86efac 38%, #22c55e 100%);
      box-shadow: 0 0 0 4px rgba(34,197,94,.16), 0 0 18px rgba(34,197,94,.45);
      animation: lockedPulse 1.15s ease-in-out infinite;
      flex: 0 0 auto;
    }
    .participant-cue.confirm { color: #fcd34d; border-color: rgba(245,158,11,.55); }
    .participant-cue.auto { color: #86efac; border-color: rgba(34,197,94,.55); }
    .participant-cue.wait { color: #cbd5e1; border-color: rgba(148,163,184,.45); }
    @keyframes lockedPulse {
      0% { transform: scale(.92); box-shadow: 0 0 0 0 rgba(34,197,94,.28), 0 0 10px rgba(34,197,94,.32); }
      70% { transform: scale(1.08); box-shadow: 0 0 0 7px rgba(34,197,94,0.0), 0 0 18px rgba(34,197,94,.48); }
      100% { transform: scale(.92); box-shadow: 0 0 0 0 rgba(34,197,94,0.0), 0 0 10px rgba(34,197,94,.32); }
    }
    .participant-question {
      font-size: 46px;
      font-weight: 800;
      line-height: 1.08;
      letter-spacing: -.02em;
      margin: 6px 0 10px 0;
    }
    .participant-subquestion {
      color: var(--muted);
      font-size: 21px;
      line-height: 1.4;
    }
    .participant-controls-card {
      max-width: 920px;
      margin: 0 auto;
    }
    .participant-minimal .controls {
      display: none !important;
    }
    .participant-minimal .participant-controls-card .section-subtitle,
    .participant-minimal .participant-controls-card #task-note {
      display: none !important;
    }
    .participant-joystick-card {
      max-width: 760px;
      margin: 0 auto;
    }
    .controller-layout {
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .participant-minimal .participant-question-card {
      max-width: 1040px;
      margin: 0 auto 22px auto;
      padding: 26px 28px;
    }
    .controller-photo {
      display: block;
      width: 100%;
      max-width: 420px;
      margin: 0;
      border-radius: 0;
      border: 0;
      background: transparent;
      object-fit: contain;
      aspect-ratio: auto;
    }
    .controller-legend {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      max-width: none;
      margin: 0;
    }
    .controller-legend .control-chip {
      min-height: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 6px;
      min-height: 84px;
      padding: 16px 14px;
      transition: border-color .15s ease, background .15s ease, box-shadow .15s ease, color .15s ease;
    }
    .controller-legend .control-chip strong {
      font-size: 22px;
      line-height: 1.1;
    }
    .controller-legend .control-chip span {
      font-size: 17px;
      line-height: 1.25;
    }
    .controller-legend .control-chip.active {
      border-color: rgba(34,197,94,.82);
      background: linear-gradient(180deg, rgba(20,83,45,.34) 0%, rgba(22,101,52,.20) 100%);
      box-shadow: 0 0 0 1px rgba(34,197,94,.24) inset, 0 0 18px rgba(34,197,94,.16);
    }
    .controller-legend .control-chip.active strong,
    .controller-legend .control-chip.active span {
      color: #dcfce7;
    }
    .operator-mode .participant-question-card {
      display: none !important;
    }
    .operator-mode .participant-banner {
      font-size: 14px;
    }
    .operator-mode .topbar {
      margin-bottom: 12px;
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
    .manual-label-layout {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
      align-items: start;
    }
    .manual-label-image {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #0b1220;
      cursor: crosshair;
    }
    .manual-label-status {
      min-height: 20px;
      color: #bfdbfe;
      font-size: 13px;
      margin-bottom: 10px;
    }
    .manual-label-controls {
      display: grid;
      gap: 10px;
    }
    .manual-label-controls select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0b1220;
      color: var(--text);
    }
    .manual-label-list {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .manual-label-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(23, 32, 51, 0.78);
    }
    .manual-label-row button {
      padding: 6px 10px;
      font-size: 12px;
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
      .manual-label-layout { grid-template-columns: 1fr; }
      .controller-layout { grid-template-columns: 1fr; }
      .controller-legend { grid-template-columns: 1fr; }
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
<body class="participant-minimal">
  <div class="page">
    <div class="title">Participant Interface</div>
    <div class="subtitle">Choose one task, confirm the required objects, then use the controller to express your intent.</div>

    <div class="topbar">
      <div class="nav-tabs" role="tablist" aria-label="Dashboard Views">
        <button class="nav-tab active" id="tab-dashboard" data-view="dashboard" type="button">Live Dashboard</button>
        <button class="nav-tab" id="tab-instructions" data-view="instructions" type="button">Instructions</button>
      </div>
      <div class="participant-banner">Nudge the joystick toward the object you want. The robot will move to pregrasp automatically; at pregrasp, press X to take manual grasp control, then press A to close.</div>
    </div>

    <div class="view active" id="view-dashboard">
      <div class="card participant-question-card">
        <div class="prompt-title">System Prompt</div>
        <div class="participant-cue wait" id="decision-cue">Wait</div>
        <div class="participant-question" id="decision-prompt">Wait for the experimenter to start the next task.</div>
        <div class="participant-subquestion" id="decision-subprompt">The interface will tell you when to move the joystick and when the system thinks it knows your target.</div>
      </div>

      <div class="layout">
        <div>
          <div class="card" id="task-status-card" style="margin-bottom: 18px;">
            <div class="section-title">Task Status</div>
            <div class="section-subtitle">This screen updates automatically when the experimenter starts the next task.</div>
            <div class="task-grid" id="task-grid"></div>
            <div class="task-actions">
              <button class="secondary" id="scan-btn">Scan Scene</button>
              <button id="rescan-btn">Quick Rescan Current Task</button>
              <button class="secondary" id="reset-btn">Reset To Scan</button>
              <button class="secondary" id="clear-sandwich-masks-btn">Clear Sandwich Masks</button>
              <button class="secondary" id="home-btn">Send Robot Home</button>
              <button class="warn" id="manual-btn" disabled hidden>Complete Manual Step</button>
            </div>
          </div>

          <div class="card" id="scene-readiness-card" style="margin-bottom: 18px;">
            <div class="readiness-head">
              <div class="readiness-title" id="scene-title">Task Status</div>
              <div class="readiness-badge idle" id="scene-status">idle</div>
            </div>
            <div class="readiness-message" id="scene-message">Run Scan Scene to record visible tags before starting a task, or use Quick Rescan Current Task after a task has started.</div>
            <div class="readiness-meta">
              <div class="readiness-stat">
                <div class="readiness-stat-label">Required Objects</div>
                <div class="readiness-stat-value" id="required-count">0</div>
              </div>
              <div class="readiness-stat">
                <div class="readiness-stat-label">Recorded For Task</div>
                <div class="readiness-stat-value" id="recorded-count">0</div>
              </div>
            </div>
            <div class="section-subtitle" id="scene-scope">No task selected.</div>
            <div class="section-subtitle">Allowed this step</div>
            <div class="readiness-list" id="allowed-list"></div>
            <div class="section-subtitle" id="allowed-note" style="margin-top: 8px;">Only objects shown under Recorded now are currently graspable.</div>
            <div class="section-subtitle">Recorded now</div>
            <div class="readiness-list" id="recorded-list"></div>
            <div class="section-subtitle" style="margin-top: 12px;">Still missing</div>
            <div class="readiness-list" id="missing-list"></div>
          </div>

          <div class="card" id="pipeline-card">
            <div class="section-title">Task Pipeline</div>
            <div class="section-subtitle">The interface automatically advances to the next automated step after a grasp completes.</div>
            <div class="pipeline" id="pipeline"></div>
          </div>

          <div class="card" id="manual-label-card" style="margin-top: 18px;">
            <div class="section-title">Sandwich Manual Labeling</div>
            <div class="section-subtitle">In scan mode, click one sandwich piece, choose its label, then save it into the candidate registry.</div>
            <div class="manual-label-layout">
              <div>
                <img class="manual-label-image" id="manual-label-image" src="/api/manual_label_image" alt="Manual sandwich labeling view">
              </div>
              <div class="manual-label-controls">
                <div class="manual-label-status" id="manual-label-status">Manual labeler not active.</div>
                <select id="manual-label-select"></select>
                <div class="task-actions">
                  <button id="manual-label-assign-btn">Assign Pending Mask</button>
                  <button class="secondary" id="manual-label-clear-btn">Clear All Labels</button>
                </div>
                <div class="section-subtitle">Labeled now</div>
                <div class="manual-label-list" id="manual-label-list"></div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <div class="card" id="guidance-card" style="margin-bottom: 18px;">
            <div class="section-title">Current Guidance</div>
            <div class="section-subtitle">Follow these prompts during the task. You do not need to monitor the internal system state.</div>
            <div class="prompt-box">
              <div class="prompt-title">What To Do Now</div>
              <div class="prompt-text" id="task-prompt">Waiting for task selection.</div>
            </div>
            <div class="prompt-box">
              <div class="prompt-title">What Happens Next</div>
              <div class="prompt-text" id="exec-prompt">Move the wrist camera to discover tags.</div>
            </div>
            <details class="operator-details">
              <summary>Operator Details</summary>
              <div class="operator-body">
                <div class="status-list">
                  <div class="status-item">
                    <div class="status-k">Current Phase</div>
                    <div class="status-v" id="current-phase">scan_workspace</div>
                  </div>
                  <div class="status-item">
                    <div class="status-k">Execution State</div>
                    <div class="status-v" id="execution-state">idle</div>
                  </div>
                  <div class="status-item">
                    <div class="status-k">Selected Label</div>
                    <div class="status-v" id="selected-label">none</div>
                  </div>
                  <div class="status-item">
                    <div class="status-k">Top Candidate</div>
                    <div class="status-v" id="top-goal">none</div>
                  </div>
                  <div class="status-item">
                    <div class="status-k">Confidence</div>
                    <div class="status-v" id="top-prob">0%</div>
                  </div>
                </div>
              </div>
            </details>
          </div>

          <div class="card participant-controls-card participant-joystick-card">
            <div class="section-title">Joystick</div>
            <div class="section-subtitle">Use the controller as shown below.</div>
            <div class="controller-layout">
              <img
                class="controller-photo"
                src="/static/participant_controller_image"
                alt="Xbox Wireless Controller image"
              >
              <div class="controller-legend">
                <div class="control-chip" data-control="x">
                  <strong>X</strong>
                  <span>Confirm / continue</span>
                </div>
                <div class="control-chip" data-control="y">
                  <strong>Y</strong>
                  <span>Cancel / reselect</span>
                </div>
                <div class="control-chip" data-control="a">
                  <strong>A</strong>
                  <span>Close fallback</span>
                </div>
                <div class="control-chip" data-control="b">
                  <strong>B</strong>
                  <span>Open at release</span>
                </div>
                <div class="control-chip" data-control="left-stick">
                  <strong>Left Stick</strong>
                  <span>Intent / XY adjust</span>
                </div>
                <div class="control-chip" data-control="right-stick">
                  <strong>Right Stick</strong>
                  <span>Vertical motion</span>
                </div>
              </div>
            </div>
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
              <span>Nudge the joystick toward the intended object. The system locks the most likely target and starts moving to pregrasp automatically.</span>
            </div>
            <div class="instruction-item">
              <strong>4. Confirm at pregrasp</strong>
              <span>During approach, keep nudging toward a different object if the target is wrong. At pregrasp, press X to take manual grasp control, use the joystick to align, then press A to close. Use Y to cancel.</span>
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
              <text class="joystick-label" x="540" y="310" text-anchor="end">A: Close fallback</text>
              <text class="joystick-note" x="540" y="328" text-anchor="end">Only if prompted</text>
              <path class="joystick-line" d="M456 214 L504 214 L534 214"/>
              <text class="joystick-label" x="540" y="210" text-anchor="end">B: Open gripper</text>
              <text class="joystick-note" x="540" y="228" text-anchor="end">At release prompt</text>
              <path class="joystick-line" d="M206 214 L86 214 L24 214"/>
              <text class="joystick-label" x="20" y="210" text-anchor="start">Left Stick: intent / XY adjust</text>
              <text class="joystick-note" x="20" y="228" text-anchor="start">Nudge target; fine tune at pregrasp</text>
              <path class="joystick-line" d="M314 258 L210 404 L24 404"/>
              <text class="joystick-label" x="20" y="400" text-anchor="start">Right Stick Vertical: Vertical motion</text>
              <text class="joystick-note" x="20" y="418" text-anchor="start">Raise or lower end-effector</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = { data: null };
    let lastDecisionCue = "";
    let audioArmed = false;
    let pendingCueKind = "";

    function probPct(v) {
      return Number.isFinite(v) ? `${Math.round(v * 100)}%` : "0%";
    }

    function armCueAudio() {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!playCueTone.ctx) {
        playCueTone.ctx = new AudioCtx();
      }
      const ctx = playCueTone.ctx;
      const resumePromise = ctx.state === "suspended" ? ctx.resume() : Promise.resolve();
      resumePromise.then(() => {
        audioArmed = true;
        if (pendingCueKind) {
          const queuedKind = pendingCueKind;
          pendingCueKind = "";
          playCueTone(queuedKind);
        }
      }).catch(() => {});
    }

    function playCueTone(kind) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!playCueTone.ctx) {
        playCueTone.ctx = new AudioCtx();
      }
      const ctx = playCueTone.ctx;
      if (!audioArmed || ctx.state === "suspended") {
        pendingCueKind = kind;
        return;
      }

      const now = ctx.currentTime;
      const master = ctx.createGain();
      master.gain.setValueAtTime(0.0001, now);
      master.connect(ctx.destination);

      const notes = kind === "confirm"
        ? [
            {freq: 880, start: 0.00, dur: 0.09, gain: 0.08},
            {freq: 1174, start: 0.11, dur: 0.12, gain: 0.10},
          ]
        : [
            {freq: 740, start: 0.00, dur: 0.10, gain: 0.08},
          ];

      notes.forEach((note) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = kind === "confirm" ? "triangle" : "sine";
        osc.frequency.setValueAtTime(note.freq, now + note.start);
        gain.gain.setValueAtTime(0.0001, now + note.start);
        gain.gain.linearRampToValueAtTime(note.gain, now + note.start + 0.015);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + note.start + note.dur);
        osc.connect(gain);
        gain.connect(master);
        osc.start(now + note.start);
        osc.stop(now + note.start + note.dur + 0.03);
      });
    }

    function installAudioArmListeners() {
      const opts = {capture: true, passive: true};
      const armOnce = () => {
        armCueAudio();
        window.removeEventListener("pointerdown", armOnce, opts);
        window.removeEventListener("keydown", armOnce, opts);
        window.removeEventListener("touchstart", armOnce, opts);
      };
      window.addEventListener("pointerdown", armOnce, opts);
      window.addEventListener("keydown", armOnce, opts);
      window.addEventListener("touchstart", armOnce, opts);
    }

    function participantTaskPrompt(data, readiness, activeStep) {
      if (!data.active_task) {
        return "Wait for the experimenter to start the next task.";
      }
      if (isAutomaticBreakfastPourActive(data, activeStep)) {
        return "Wait for the robot to finish the automatic pour sequence.";
      }
      if (readiness.status !== "ready") {
        return "The system is checking the required objects for this task. Wait while the experimenter finishes setup.";
      }
      if (activeStep) {
        return activeStep.description || `Start the step: ${activeStep.title}.`;
      }
      return "Follow the current task instructions.";
    }

    function isDestinationStep(activeStep) {
      if (!activeStep) return false;
      const idText = String(activeStep.id || "").toLowerCase();
      const titleText = String(activeStep.title || "").toLowerCase();
      const descText = String(activeStep.description || "").toLowerCase();
      return (
        idText.includes("_container")
        || idText.includes("destination")
        || idText.includes("place_")
        || titleText.includes("destination")
        || titleText.includes("bowl")
        || titleText.includes("placement")
        || descText.includes("destination")
        || descText.includes("bowl")
        || descText.includes("drop")
        || descText.includes("release")
      );
    }

    function titleCaseObjectName(name) {
      if (!name) return "";
      return String(name).replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
    }

    function nextTargetPrompt(activeStep) {
      if (!activeStep) {
        return {
          title: "Move Toward Your Next Target",
          subtitle: "Move clearly toward the next target to continue."
        };
      }
      const stepId = String(activeStep.id || "").toLowerCase();
      if (stepId === "select_breakfast_ingredient") {
        return {
          title: "Move Toward A Breakfast Ingredient",
          subtitle: "Nudge toward one cereal box or the chocolate powder. The robot will start pregrasp when the intent is stable."
        };
      }
      if (stepId === "select_breakfast_milk") {
        return {
          title: "Move Toward A Milk Carton",
          subtitle: "Nudge toward one milk carton. The robot will start pregrasp when the intent is stable."
        };
      }
      if (stepId === "select_breakfast_ingredient_container" || stepId === "select_breakfast_milk_container") {
        return {
          title: "Move Toward The Bowl",
          subtitle: "Guide the held item toward the bowl, then confirm to move above it."
        };
      }
      if (stepId === "select_sandwich_item") {
        return {
          title: "Move Toward Your Next Sandwich Piece",
          subtitle: "Nudge toward the next sandwich piece. The robot will start pregrasp when the intent is stable."
        };
      }
      if (stepId === "select_sort_object") {
        return {
          title: "Move Toward The Next Item",
          subtitle: "Move clearly toward the next item you want to sort."
        };
      }
      if (stepId === "select_lego_brick") {
        return {
          title: "Move Toward The Next LEGO Brick",
          subtitle: "Move clearly toward the next LEGO brick to continue."
        };
      }
      if (isDestinationStep(activeStep)) {
        return {
          title: "Move Toward Your Target",
          subtitle: "Move clearly toward the target where you want to place the item."
        };
      }
      return {
        title: "Move Toward Your Next Target",
        subtitle: "Nudge toward the next target. The robot will start pregrasp when the intent is stable."
      };
    }

    function completionTransitionStep(data, activeStep, execState) {
      const activeTask = data.active_task;
      if (!activeTask || !Array.isArray(activeTask.steps) || !activeTask.steps.length || !activeStep) {
        return activeStep;
      }
      const stepIndex = activeTask.steps.findIndex((step) => String(step.id || "") === String(activeStep.id || ""));
      if (stepIndex < 0) {
        return activeStep;
      }
      const taskId = String(activeTask.id || "").toLowerCase();
      const stepId = String(activeStep.id || "").toLowerCase();
      if (
        taskId === "make_breakfast"
        && (stepId === "pour_breakfast_ingredient" || stepId === "pour_breakfast_milk")
        && execState.includes("grasp_complete")
      ) {
        return activeStep;
      }
      if (execState.includes("release_complete")) {
        if (taskId === "make_sandwich" || taskId === "sorting" || taskId === "lego_sorting") {
          return activeTask.steps[0] || activeStep;
        }
        const nextIndex = stepIndex + 1;
        return activeTask.steps[nextIndex] || activeStep;
      }
      if (execState.includes("grasp_complete")) {
        const nextIndex = stepIndex + 1;
        return activeTask.steps[nextIndex] || activeStep;
      }
      return activeStep;
    }

    function isBreakfastPourStep(step) {
      if (!step) return false;
      const stepId = String(step.id || "").toLowerCase();
      return stepId === "pour_breakfast_ingredient" || stepId === "pour_breakfast_milk";
    }

    function isAutomaticBreakfastPourActive(data, activeStep) {
      if (!isBreakfastPourStep(activeStep)) {
        return false;
      }
      if (Boolean(data.breakfast_pour_active)) {
        return true;
      }
      const carriedLabel = String(data.active_breakfast_item_label || "").trim();
      return carriedLabel.length > 0;
    }

    function isExecutionLocked(execState) {
      return execState.includes("wait_pregrasp_confirm")
        || execState.includes("wait_grasp_confirm")
        || execState.includes("wait_close_a")
        || execState.includes("wait_open_b")
        || execState.includes("exec_")
        || execState.includes("grasp_complete")
        || execState.includes("release_complete");
    }

    function displayedTarget(data) {
      const execState = String(data.execution_state || "").toLowerCase();
      const selectedLabel = String(data.selected_grasp_label || "").trim();
      const selectedObjectName = titleCaseObjectName(data.selected_grasp_object_name || "");
      const topGoalLabel = String(data.top_goal_label || "").trim();
      const topGoalObjectName = titleCaseObjectName(data.top_goal_object_name || "");
      if (selectedLabel && isExecutionLocked(execState)) {
        return {
          label: selectedObjectName || selectedLabel,
          probabilityText: "Locked"
        };
      }
      return {
        label: topGoalObjectName || topGoalLabel || "none",
        probabilityText: probPct(data.top_probability)
      };
    }

    function participantDecisionPrompt(data, readiness) {
      const activeStep = data.active_task ? data.active_task.steps.find((step) => step.status === "active") : null;
      const destinationStep = isDestinationStep(activeStep);
      const confidence = Number.isFinite(data.top_probability) ? data.top_probability : 0;
      const takeoverThreshold = Number.isFinite(data.participant_takeover_threshold)
        ? data.participant_takeover_threshold
        : 0.6;
      const execState = String(data.execution_state || "").toLowerCase();
      const confirmationPrompt = String(data.confirmation_prompt || "").trim().toLowerCase();
      const selectedLabel = String(data.selected_grasp_label || "").trim();
      const selectedObjectName = titleCaseObjectName(data.selected_grasp_object_name || "");
      const topGoalObjectName = titleCaseObjectName(data.top_goal_object_name || "");
      const fallbackTopGoalName = topGoalObjectName || (data.top_goal_label ? `Target ${data.top_goal_label}` : "");
      const confirmedObjectName = selectedObjectName || topGoalObjectName;
      const hasLockedSelection = selectedLabel.length > 0;
      const objectName = hasLockedSelection
        ? confirmedObjectName
        : fallbackTopGoalName;
      const selectionReadyFromPrompt = confirmationPrompt.includes("press x to execute grasp")
        || confirmationPrompt.includes("press x to move above the selected container");
      const selectionReady = hasLockedSelection || Boolean(data.selection_ready) || (confidence >= takeoverThreshold) || selectionReadyFromPrompt;
      if (!data.active_task) {
        return {
          cue: "wait",
          cueLabel: "Wait",
          title: "Wait For The Next Task",
          subtitle: "The experimenter will start the task. Do not move yet."
        };
      }
      if (isAutomaticBreakfastPourActive(data, activeStep)) {
        return {
          cue: "auto",
          cueLabel: "Robot Pouring",
          title: "Wait For The Robot To Finish",
          subtitle: "The robot is lifting, moving to the bowl, pouring, and placing the item back. Keep clear."
        };
      }
      if (readiness.status !== "ready") {
        return {
          cue: "wait",
          cueLabel: "Wait",
          title: "Wait For The Next Step",
          subtitle: "The robot or operator is preparing the next step. Hold still and wait."
        };
      }
      if (
        !selectionReady
        && !execState.includes("wait_pregrasp_confirm")
        && !execState.includes("wait_grasp_confirm")
        && !execState.includes("wait_close_a")
        && !execState.includes("wait_open_b")
        && !execState.includes("exec_")
        && !execState.includes("grasp")
        && !execState.includes("pregrasp")
      ) {
        return {
          cue: "move",
          cueLabel: "Move Clearly",
          title: "Move Toward Your Target",
          subtitle: destinationStep
            ? "Use clear joystick motion to show your intent. Keep moving a little farther until the placement target is locked."
            : "Nudge the joystick toward the object you want. When the intent is stable, the robot will start moving to pregrasp automatically."
        };
      }
      if (execState.includes("wait_pregrasp_confirm") && hasLockedSelection && selectionReady && objectName) {
        return {
          cue: "locked",
          cueLabel: destinationStep ? "Placement Locked" : "Target Locked",
          title: destinationStep ? `Are You Going To Place At ${objectName}?` : `Are You Going To Grasp ${objectName}?`,
          subtitle: destinationStep
            ? "Press X to move above the placement target. Press Y to cancel."
            : "Target is loaded. The robot should start pregrasp automatically; press X only if it is waiting. Press Y to cancel."
        };
      }
      if (execState.includes("wait_grasp_confirm")) {
        return {
          cue: "confirm",
          cueLabel: "At Pregrasp",
          title: objectName ? `Is The Gripper Centered Over ${objectName}?` : "Is The Gripper Centered Over The Target?",
          subtitle: "Press X to take manual grasp control. Then use the joystick to align/lower and press A to close the gripper. Press Y to cancel."
        };
      }
      if (execState.includes("wait_close_a")) {
        return {
          cue: "confirm",
          cueLabel: "Manual Grasp",
          title: objectName ? `Ready To Close On ${objectName}?` : "Ready To Close The Gripper?",
          subtitle: "Use the joystick for final alignment, then press A to close the gripper."
        };
      }
      if (execState.includes("wait_open_b")) {
        return {
          cue: "confirm",
          cueLabel: "At Release",
          title: objectName ? `Ready To Release At ${objectName}?` : "Ready To Release?",
          subtitle: "You can make a slight height adjustment now. Press B to open gripper."
        };
      }
      if (isBreakfastPourStep(activeStep) && execState.includes("grasp_complete")) {
        return {
          cue: "auto",
          cueLabel: "Robot Pouring",
          title: "Automatic Pour In Progress",
          subtitle: "The robot is lifting, moving to the bowl, pouring, and placing the item back. Keep clear."
        };
      }
      if (execState.includes("grasp_complete") || execState.includes("release_complete")) {
        const transitionStep = completionTransitionStep(data, activeStep, execState);
        const nextPrompt = nextTargetPrompt(transitionStep);
        return {
          cue: "move",
          cueLabel: "Move Clearly",
          title: nextPrompt.title,
          subtitle: nextPrompt.subtitle
        };
      }
      if (!hasLockedSelection && objectName && selectionReady) {
        if (destinationStep) {
          return {
            cue: "locked",
            cueLabel: "Placement Locked",
            title: objectName ? `Are You Going To Place At ${objectName}?` : "Are You Going To Place At This Destination?",
            subtitle: "Press X to confirm. Press Y to cancel."
          };
        }
        return {
          cue: "locked",
          cueLabel: "Target Locked",
          title: objectName
            ? `Are You Going To Grasp ${objectName}?`
            : "Are You Going To Grasp This Target?",
          subtitle: "If this is correct, release the joystick and let the robot start pregrasp. If it is wrong, nudge toward the target you want."
        };
      }
	      if (execState.includes("retreat_before_pregrasp")) {
	        return {
	          cue: "auto",
	          cueLabel: "Retargeting",
	          title: "Moving Away Before The New Target",
	          subtitle: "The robot is backing up before approaching the retargeted object. Keep clear unless you need to cancel."
	        };
	      }
	      if (execState.includes("exec_pregrasp")) {
	        return {
	          cue: "auto",
	          cueLabel: "Robot Moving",
          title: objectName ? `Moving To Pregrasp For ${objectName}` : "Moving To Pregrasp",
	          subtitle: "If the target is wrong, nudge clearly toward the correct object. If the current pose is a good pregrasp, press X to take manual grasp control."
        };
      }
      if (execState.includes("exec_grasp")) {
        return {
          cue: "auto",
          cueLabel: "Grasping",
          title: objectName ? `Touching Down On ${objectName}` : "Touching Down",
          subtitle: "The robot is executing the final grasp. Do not retarget now; press Y only if you need to cancel."
        };
      }
      if (
        execState.includes("exec_")
        || (
          (execState.includes("grasp") || execState.includes("pregrasp"))
          && !execState.includes("grasp_complete")
          && !execState.includes("release_complete")
          && !execState.includes("wait_pregrasp_confirm")
        )
      ) {
        return {
          cue: "auto",
          cueLabel: "Robot Moving",
          title: "Let The Robot Finish",
          subtitle: destinationStep
            ? "The robot is carrying the item toward the destination. Keep clear unless you need to cancel."
            : "The robot is executing the assistance motion. Keep clear unless you need to cancel."
        };
      }
      return {
        cue: "move",
        cueLabel: "Move Clearly",
        title: "Move Toward Your Target",
        subtitle: destinationStep
          ? "Use clear joystick motion to show your intent. Keep moving a little farther until the placement target is locked."
          : "Nudge the joystick toward the object you want. When the intent is stable, the robot will start moving to pregrasp automatically."
      };
    }

    function participantExecutionPrompt(data, readiness) {
      const execState = String(data.execution_state || "").toLowerCase();
      const activeStep = data.active_task ? data.active_task.steps.find((step) => step.status === "active") : null;
      const destinationStep = isDestinationStep(activeStep);
      const selectedLabel = String(data.selected_grasp_label || "").trim();
      const objectName = titleCaseObjectName(data.selected_grasp_object_name || data.top_goal_object_name || "")
        || (data.top_goal_label ? `Target ${data.top_goal_label}` : "");
      const confidence = Number.isFinite(data.top_probability) ? data.top_probability : 0;
      const takeoverThreshold = Number.isFinite(data.participant_takeover_threshold)
        ? data.participant_takeover_threshold
        : 0.6;
      const confirmationPrompt = String(data.confirmation_prompt || "").trim();
      const confirmationPromptLower = confirmationPrompt.toLowerCase();
      const selectionReadyFromPrompt = confirmationPromptLower.includes("press x to execute grasp")
        || confirmationPromptLower.includes("press x to move above the selected container");
      const selectionReady = selectedLabel.length > 0 || Boolean(data.selection_ready) || (confidence >= takeoverThreshold) || selectionReadyFromPrompt;
      if (!data.active_task) {
        return "Do not move yet.";
      }
      if (isAutomaticBreakfastPourActive(data, activeStep)) {
        return "Wait for the robot to finish the automatic pour sequence.";
      }
      if (readiness.status !== "ready") {
        return "Wait for the next step. The robot or operator is preparing the task.";
      }
      if (
        !selectionReady
        && !execState.includes("wait_pregrasp_confirm")
        && !execState.includes("wait_grasp_confirm")
        && !execState.includes("wait_close_a")
        && !execState.includes("wait_open_b")
        && !execState.includes("exec_")
        && !execState.includes("grasp")
        && !execState.includes("pregrasp")
      ) {
        return destinationStep
          ? "Move clearly toward the target where you want to place the item. A small extra motion is required before automatic placement can take over."
          : "Nudge toward the target you want. The system will choose the stable intent and start moving to pregrasp automatically.";
      }
      if (execState.includes("wait") && execState.includes("target")) {
        return "Move the wrist camera until the required objects are visible.";
      }
      if (!selectedLabel && selectionReadyFromPrompt && !execState.includes("wait_pregrasp_confirm")) {
        return destinationStep
          ? (objectName
              ? `Are you going to place at ${objectName}? Press X to confirm. Press Y to cancel.`
              : "Are you going to place at this destination? Press X to confirm. Press Y to cancel.")
          : (objectName
              ? `Are you going to grasp ${objectName}? Press X to confirm. Press Y to cancel.`
              : "Are you going to grasp this target? Press X to confirm. Press Y to cancel.");
      }
      if (execState.includes("wait_pregrasp_confirm") && selectedLabel && selectionReady) {
        return destinationStep
          ? "This placement target is locked. Press X to move above it. Press Y to cancel."
          : "This grasp target is loaded. The robot should move to pregrasp automatically; press X only if it is waiting. Press Y to cancel.";
      }
      if (execState.includes("wait_grasp_confirm")) {
        return objectName
          ? `At pregrasp for ${objectName}. Press X to take manual grasp control, then use the joystick and press A to close. Press Y to cancel.`
          : "At pregrasp. Press X to take manual grasp control, then use the joystick and press A to close. Press Y to cancel.";
      }
      if (execState.includes("wait_close_a")) {
        return objectName
          ? `Manual grasp control for ${objectName}. Use the joystick to align/lower, then press A to close gripper now.`
          : "Manual grasp control. Use the joystick to align/lower, then press A to close gripper now.";
      }
      if (execState.includes("wait_open_b")) {
        return objectName
          ? `At prerelease for ${objectName}. You can make a slight height adjustment now. Press B to open gripper.`
          : "At prerelease. You can make a slight height adjustment now. Press B to open gripper.";
      }
      if (isBreakfastPourStep(activeStep) && execState.includes("grasp_complete")) {
        return "Automatic pour in progress. The robot is moving to the bowl, pouring, and placing the item back.";
      }
      if (execState.includes("grasp_complete") || execState.includes("release_complete")) {
        return nextTargetPrompt(completionTransitionStep(data, activeStep, execState)).subtitle;
      }
      if (destinationStep && execState.includes("grasp_complete")) {
        return "Placement finished. Wait for the next instruction.";
      }
      if (destinationStep && execState.includes("idle")) {
        if (selectionReady && objectName) {
          return `Are you going to place at ${objectName}? Press X to confirm. Press Y to cancel.`;
        }
        return "Move clearly toward the target where you want to place the item. A small extra motion is required before automatic placement can take over.";
      }
	      if (
	        execState.includes("retreat_before_pregrasp")
	      ) {
	        return "Retargeting is active. The robot is backing up before moving toward the new object.";
	      }
	      if (
	        execState.includes("exec_pregrasp")
	      ) {
	        return "The robot is moving to pregrasp. If the target is wrong, nudge clearly toward the correct object. If the current pose is good, press X to use it as pregrasp and take manual grasp control.";
      }
      if (
        execState.includes("exec_grasp")
      ) {
        return "The robot is touching down and grasping. Do not retarget during this final grasp motion.";
      }
      if (
        execState.includes("grasp")
        && !execState.includes("grasp_complete")
        && !execState.includes("release_complete")
        && !execState.includes("wait_pregrasp_confirm")
      ) {
        return destinationStep
          ? "The robot is moving to the destination. Be ready to release with B."
          : "The robot is executing the grasp motion.";
      }
      if (execState === "idle") {
        if (!selectionReady) {
          return destinationStep
            ? "Move clearly toward the target where you want to place the item until the placement target is ready."
            : "Nudge toward the target you want. The system will start automatic pregrasp when the intent is stable.";
        }
        return destinationStep
          ? (objectName
              ? `Are you going to place at ${objectName}? Press X to confirm. Press Y to cancel.`
              : "Show the placement target with clear joystick motion. Keep moving a little farther to unlock automatic placement.")
          : "If the shown target is correct, release the joystick and let the robot move to pregrasp. If it is wrong, nudge toward the intended object.";
      }
      return "Keep your motion clear and follow the current prompt.";
    }

    function activeControlKey(data) {
      const execState = String(data.execution_state || "").toLowerCase();
      if (execState.includes("exec_pregrasp") || execState.includes("retreat_before_pregrasp")) return "left-stick";
      if (execState.includes("wait_pregrasp_confirm")) return "x";
      if (execState.includes("wait_grasp_confirm")) return "x";
      if (execState.includes("wait_close_a")) return "a";
      if (execState.includes("wait_open_b")) return "b";
      if (execState === "idle" || execState.includes("wait") || execState.includes("target")) return "left-stick";
      return "";
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
      const activeTask = data.active_task;
      const isOperator = document.body.classList.contains("operator-mode");
      if (!isOperator) {
        if (!activeTask) {
          root.innerHTML = "";
          return;
        }
        root.innerHTML = `
          <div class="task-card active">
            <div class="task-title">${activeTask.display_name}</div>
            <div class="task-desc">${activeTask.description || ""}</div>
          </div>
        `;
        return;
      }

      if (!tasks.length) {
        root.innerHTML = "";
        return;
      }
      root.innerHTML = tasks.map((task) => {
        const activeClass = activeTask && activeTask.id === task.id ? "active" : "";
        const buttonLabel = activeTask && activeTask.id === task.id ? "Running" : "Start Task";
        const disabled = activeTask && activeTask.id === task.id ? "disabled" : "";
        return `
          <div class="task-card ${activeClass}">
            <div class="task-title">${task.display_name}</div>
            <div class="task-desc">${task.description || ""}</div>
            <div class="task-actions">
              <button type="button" data-task-id="${task.id}" class="start-task-btn" ${disabled}>${buttonLabel}</button>
            </div>
          </div>
        `;
      }).join("");

      root.querySelectorAll(".start-task-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const taskId = String(btn.dataset.taskId || "").trim();
          if (!taskId) return;
          await postJSON("/api/start_task", {task_id: taskId});
          await refreshState();
        });
      });
    }

    function renderPipeline(data) {
      const root = document.getElementById("pipeline");
      const activeTask = data.active_task;
      if (!activeTask) {
        root.innerHTML = '<div class="step"><div class="step-desc">No task selected. Choose a task to begin the guided pipeline.</div></div>';
        const manualBtn = document.getElementById("manual-btn");
        manualBtn.disabled = true;
        manualBtn.hidden = true;
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
      const hasManualStep = activeTask.steps.some((step) => step.manual);
      const manualBtn = document.getElementById("manual-btn");
      manualBtn.hidden = !hasManualStep;
      manualBtn.disabled = !(activeStep && activeStep.manual);
    }

    function renderReadiness(data) {
      const readiness = data.scene_readiness || {};
      const activeTask = data.active_task;
      const activeStep = activeTask ? activeTask.steps.find((step) => step.status === "active") : null;
      const automaticBreakfastPour = isAutomaticBreakfastPourActive(data, activeStep);
      const status = readiness.status || "idle";
      const statusEl = document.getElementById("scene-status");
      if (automaticBreakfastPour) {
        statusEl.textContent = "robot busy";
        statusEl.className = "readiness-badge ready";
        document.getElementById("scene-title").textContent = "Robot Execution";
        document.getElementById("scene-message").textContent = "Automatic pouring is in progress. Wait for the robot to finish.";
        document.getElementById("required-count").textContent = "1";
        document.getElementById("recorded-count").textContent = "1";
        document.getElementById("scene-scope").textContent = "Current breakfast item is already grasped and being poured.";
        document.getElementById("allowed-note").textContent = "The robot is executing the automatic pour sequence. Keep clear until it finishes.";
      } else {
        statusEl.textContent = status.replaceAll("_", " ");
        statusEl.className = `readiness-badge ${status}`;
        document.getElementById("scene-title").textContent = readiness.title || "Task Status";
        document.getElementById("scene-message").textContent = readiness.message || "Scan the scene or start a task.";
        document.getElementById("required-count").textContent = String(readiness.required_count || 0);
        document.getElementById("recorded-count").textContent = String(readiness.recorded_count || 0);
        document.getElementById("scene-scope").textContent = readiness.scope_label || "No task selected.";
        document.getElementById("allowed-note").textContent = readiness.allowed_note
          || "Only objects shown under Recorded now are currently graspable.";
      }

      const allowedRoot = document.getElementById("allowed-list");
      const recordedRoot = document.getElementById("recorded-list");
      const missingRoot = document.getElementById("missing-list");
      const allowedObjects = automaticBreakfastPour ? [] : (readiness.allowed_objects || []);
      const recordedObjects = automaticBreakfastPour ? [] : (readiness.recorded_objects || []);
      const missingObjects = automaticBreakfastPour ? [] : (readiness.missing_objects || []);

      allowedRoot.innerHTML = allowedObjects.length
        ? allowedObjects.map((item) => `<div class="readiness-chip">${item.object_name || item.label || "unknown"}</div>`).join("")
        : '<div class="notice">No task-specific object list available.</div>';
      recordedRoot.innerHTML = recordedObjects.length
        ? recordedObjects.map((item) => `<div class="readiness-chip recorded">${item.object_name || item.label || "unknown"}</div>`).join("")
        : '<div class="notice">No recorded task objects yet.</div>';
      missingRoot.innerHTML = missingObjects.length
        ? missingObjects.map((item) => `<div class="readiness-chip missing">${item.object_name || item.label || "unknown"}</div>`).join("")
        : '<div class="notice">Nothing missing.</div>';

      const rescanBtn = document.getElementById("rescan-btn");
      rescanBtn.disabled = !data.active_task;
      rescanBtn.textContent = readiness.rescan_active ? "Rescan In Progress" : "Quick Rescan Current Task";
    }

    function renderManualLabeling(data) {
      const manual = data.manual_labeler || {};
      const card = document.getElementById("manual-label-card");
      card.style.display = "block";
      document.getElementById("manual-label-status").textContent = manual.enabled
        ? (manual.status || "Click an object to create a pending mask.")
        : "Manual labeler not active. Restart the sandwich launch and hard-refresh this page.";
      const select = document.getElementById("manual-label-select");
      const available = manual.available_labels || [];
      const currentValue = select.value;
      select.innerHTML = available.map((name) => `<option value="${name}">${name}</option>`).join("");
      if (available.includes(currentValue)) {
        select.value = currentValue;
      }
      document.getElementById("manual-label-assign-btn").disabled = !manual.enabled || !manual.pending_ready;
      document.getElementById("manual-label-clear-btn").disabled = !manual.enabled;
      select.disabled = !manual.enabled || available.length === 0;
      const listRoot = document.getElementById("manual-label-list");
      const labeled = manual.labeled_objects || [];
      listRoot.innerHTML = labeled.length
        ? labeled.map((item) => `
            <div class="manual-label-row">
              <span>${item.object_name} (${item.candidate_id})</span>
              <button class="secondary manual-remove-btn" data-object-name="${item.object_name}" type="button">Remove</button>
            </div>`).join("")
        : '<div class="notice">No labeled sandwich objects yet.</div>';
      listRoot.querySelectorAll(".manual-remove-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await postJSON("/api/manual_label_remove", {object_name: btn.dataset.objectName || ""});
          await refreshState();
        });
      });

      const clearMasksBtn = document.getElementById("clear-sandwich-masks-btn");
      if (clearMasksBtn) {
        clearMasksBtn.disabled = !manual.enabled;
      }
    }

    function render(data) {
      const activeTask = data.active_task;
      const activeStep = activeTask ? activeTask.steps.find((step) => step.status === "active") : null;
      const readiness = data.scene_readiness || {};
      const decision = participantDecisionPrompt(data, readiness);
      const questionCard = document.querySelector(".participant-question-card");
      questionCard.classList.remove("move", "locked", "confirm", "auto", "wait");
      questionCard.classList.add(decision.cue || "wait");
      const cueEl = document.getElementById("decision-cue");
      cueEl.className = `participant-cue ${decision.cue || "wait"}`;
      cueEl.textContent = decision.cueLabel || "Wait";
      const cueKey = String(decision.cue || "wait");
      if (cueKey !== lastDecisionCue) {
        if (cueKey === "locked") {
          playCueTone("locked");
        } else if (cueKey === "confirm") {
          playCueTone("confirm");
        }
        lastDecisionCue = cueKey;
      }
      document.getElementById("decision-prompt").textContent = decision.title;
      document.getElementById("decision-subprompt").textContent = decision.subtitle;
      document.getElementById("current-phase").textContent = data.current_phase || "scan_workspace";
      const displayTarget = displayedTarget(data);
      document.getElementById("top-goal").textContent = displayTarget.label;
      document.getElementById("top-prob").textContent = displayTarget.probabilityText;
      document.getElementById("task-prompt").textContent = participantTaskPrompt(data, readiness, activeStep);
      document.getElementById("exec-prompt").textContent = participantExecutionPrompt(data, readiness);
      document.getElementById("selected-label").textContent = data.selected_grasp_label || "none";
      document.getElementById("execution-state").textContent = data.execution_state || "idle";
      const activeControl = activeControlKey(data);
      document.querySelectorAll(".controller-legend .control-chip").forEach((chip) => {
        chip.classList.toggle("active", String(chip.dataset.control || "") === activeControl);
      });

      renderTasks(data);
      renderReadiness(data);
      renderPipeline(data);
      renderManualLabeling(data);
    }

    let refreshTimer = null;
    let refreshInFlight = false;

    function desiredRefreshMs() {
      const data = state.data || {};
      const hasActiveTask = Boolean(data.active_task);
      const execState = String(data.execution_state || "").toLowerCase();
      if (execState.includes("exec_") || execState.includes("wait_") || execState.includes("grasp_complete")) {
        return 120;
      }
      if (hasActiveTask) {
        return 180;
      }
      return 700;
    }

    function scheduleRefresh(delayMs) {
      if (refreshTimer) {
        clearTimeout(refreshTimer);
      }
      refreshTimer = setTimeout(() => {
        refreshState();
      }, Math.max(50, Number(delayMs) || desiredRefreshMs()));
    }

    async function refreshState() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
        const res = await fetch(`/api/state?ts=${Date.now()}`, {cache: "no-store"});
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.data = await res.json();
        render(state.data);
      } catch (err) {
        console.error("dashboard refresh failed", err);
      } finally {
        refreshInFlight = false;
        scheduleRefresh(desiredRefreshMs());
      }
    }

    document.getElementById("scan-btn").addEventListener("click", async () => {
      await postJSON("/api/scan_scene", {});
      await refreshState();
    });

    document.getElementById("reset-btn").addEventListener("click", async () => {
      await postJSON("/api/reset_task", {});
      await refreshState();
    });

    document.getElementById("clear-sandwich-masks-btn").addEventListener("click", async () => {
      await postJSON("/api/manual_label_clear", {});
      await refreshState();
    });

    document.getElementById("rescan-btn").addEventListener("click", async () => {
      await postJSON("/api/quick_rescan", {});
      await refreshState();
    });

    document.getElementById("manual-btn").addEventListener("click", async () => {
      await postJSON("/api/manual_advance", {});
      await refreshState();
    });

    document.getElementById("home-btn").addEventListener("click", async () => {
      await postJSON("/api/send_home", {});
      await refreshState();
    });

    document.getElementById("manual-label-image").addEventListener("click", async (event) => {
      const img = event.currentTarget;
      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / Math.max(rect.width, 1);
      const scaleY = img.naturalHeight / Math.max(rect.height, 1);
      const u = Math.round((event.clientX - rect.left) * scaleX);
      const v = Math.round((event.clientY - rect.top) * scaleY);
      await postJSON("/api/manual_label_click", {u, v});
      await refreshState();
    });

    document.getElementById("manual-label-assign-btn").addEventListener("click", async () => {
      const objectName = document.getElementById("manual-label-select").value || "";
      await postJSON("/api/manual_label_assign", {object_name: objectName});
      await refreshState();
    });

    document.getElementById("manual-label-clear-btn").addEventListener("click", async () => {
      await postJSON("/api/manual_label_clear", {});
      await refreshState();
    });

    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    });

    window.addEventListener("hashchange", () => {
      setView(window.location.hash === "#instructions" ? "instructions" : "dashboard");
    });

    installAudioArmListeners();
    setView(window.location.hash === "#instructions" ? "instructions" : "dashboard");
    refreshState();
    setInterval(() => {
      const img = document.getElementById("manual-label-image");
      const manual = (state.data && state.data.manual_labeler) || {};
      if (!manual.enabled) return;
      img.src = `/api/manual_label_image?ts=${Date.now()}`;
    }, 800);
  </script>
</body>
</html>
"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _render_dashboard_html(mode):
    mode = str(mode).strip().lower()
    if mode == "operator":
        html = HTML_PAGE.replace('body class="participant-minimal"', 'body class="operator-mode"', 1)
        html = html.replace("<title>Participant Interface</title>", "<title>Experimenter Interface</title>", 1)
        html = html.replace(
            '<div class="title">Participant Interface</div>',
            '<div class="title">Experimenter Interface</div>',
            1,
        )
        html = html.replace(
            '<div class="subtitle">Choose one task, confirm the required objects, then use the controller to express your intent.</div>',
            '<div class="subtitle">Start tasks, monitor system state, and manage resets or rescans during the study.</div>',
            1,
        )
        html = html.replace(
            '<div class="participant-banner">Watch the prompt below. Use the joystick only to move the robot and respond when the system asks about a target object.</div>',
            '<div class="participant-banner">Use this page to start tasks, monitor readiness, and manage the study flow. The participant should use the separate participant page.</div>',
            1,
        )
        return html
    return HTML_PAGE


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
        self.participant_controller_image = os.path.expanduser(
            rospy.get_param(
                "~participant_controller_image",
                "/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/assets/xbox_controller_edgecut.png",
            )
        )
        self.host = str(rospy.get_param("~host", "127.0.0.1")).strip()
        self.port = int(rospy.get_param("~port", 8766))
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.task_prompt_topic = str(rospy.get_param("~task_prompt_topic", "/task_context/prompt")).strip()
        self.execution_prompt_topic = str(rospy.get_param("~execution_prompt_topic", "/apriltag_executor/prompt")).strip()
        self.confirmation_prompt_topic = str(
            rospy.get_param("~confirmation_prompt_topic", "/apriltag_intent_inference/confirmation_prompt")
        ).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.ee_pose_goal_topic = str(rospy.get_param("~ee_pose_goal_topic", "/relaxed_ik/ee_pose_goals")).strip()
        self.relaxed_ik_reset_topic = str(rospy.get_param("~relaxed_ik_reset_topic", "/relaxed_ik/reset")).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_prob_topic = str(rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")).strip()
        self.selection_ready_topic = str(
            rospy.get_param("~selection_ready_topic", "/intent_inference/selection_ready")
        ).strip()
        self.participant_takeover_threshold = float(rospy.get_param("~participant_takeover_threshold", 0.6))
        self.distribution_topic = str(rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")).strip()
        self.candidates_topic = str(rospy.get_param("~candidates_topic", "/apriltag_grasp_registry/detections")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.carried_grasp_label_topic = str(
            rospy.get_param("~carried_grasp_label_topic", "/shared_autonomy/carried_grasp_label")
        ).strip()
        self.trial_context_topic = str(
            rospy.get_param("~trial_context_topic", "/user_study/trial_context")
        ).strip()
        self.study_event_topic = str(rospy.get_param("~study_event_topic", "/user_study/events")).strip()
        self.manual_label_command_topic = str(rospy.get_param("~manual_label_command_topic", "/sam_manual_labeler/command")).strip()
        self.manual_label_state_topic = str(rospy.get_param("~manual_label_state_topic", "/sam_manual_labeler/state")).strip()
        self.manual_label_debug_image_topic = str(rospy.get_param("~manual_label_debug_image_topic", "/sam_manual_labeler/debug_image")).strip()
        self.initial_reset_command = str(rospy.get_param("~initial_reset_command", "scan_workspace")).strip()
        self.home_pause_topic = str(rospy.get_param("~home_pause_topic", "/shared_autonomy/home_motion_active")).strip()
        self.home_limb = str(rospy.get_param("~home_limb", "right")).strip()
        self.home_timeout = float(rospy.get_param("~home_timeout", 20.0))
        self.home_speed = float(rospy.get_param("~home_speed", 0.2))
        self.home_retreat_pose_topic = str(
            rospy.get_param("~home_retreat_pose_topic", "/tag_grasp_demo/pregrasp_pose")
        ).strip()
        self.home_retreat_timeout = float(rospy.get_param("~home_retreat_timeout", 2.0))
        self.home_retreat_position_tolerance = float(rospy.get_param("~home_retreat_position_tolerance", 0.02))
        self.target_source_registry_label = str(
            rospy.get_param("~target_source_registry_label", "recorded_apriltag_registry")
        ).strip()
        self.target_source_realtime_lego_label = str(
            rospy.get_param("~target_source_realtime_lego_label", "realtime_sam_mask")
        ).strip()
        self.target_source_hybrid_label = str(
            rospy.get_param("~target_source_hybrid_label", "hybrid_candidate_mux")
        ).strip()
        self.session_id = str(rospy.get_param("~session_id", datetime.now().strftime("session_%Y%m%d_%H%M%S"))).strip()
        self.participant_id = str(rospy.get_param("~participant_id", "")).strip()
        self.condition_id = str(rospy.get_param("~condition_id", "")).strip()
        self.block_id = str(rospy.get_param("~block_id", "")).strip()
        default_log_dir = os.path.join(package_root, "logs")
        self.enable_probability_logging = bool(rospy.get_param("~enable_probability_logging", True))
        self.probability_log_dir = os.path.expanduser(
            rospy.get_param("~probability_log_dir", default_log_dir)
        )

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.tasks = self._load_tasks()
        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()
        self.active_task_id = None
        self.active_step_index = None
        self.current_phase = self.initial_reset_command
        self.task_prompt = "Waiting for task selection."
        self.execution_prompt = "Move the joystick to indicate intent."
        self.confirmation_prompt = ""
        self.execution_state = "idle"
        self.top_goal_label = ""
        self.top_probability = 0.0
        self.selection_ready = False
        self.selected_grasp_label = ""
        self.latest_ee_pose = None
        self.allowed_tag_ids = set()
        self.latest_candidate_labels = []
        self.last_distribution = []
        self.last_distribution_stamp = ""
        self.probability_log_path = self._make_probability_log_path() if self.enable_probability_logging else ""
        self.rescan_active = False
        self.home_hold_active = False
        self.latest_home_retreat_pose = None
        self.active_sandwich_item_label = ""
        self.active_breakfast_item_label = ""
        self.breakfast_pour_active = False
        self.manual_label_state = {"enabled": False}
        self.manual_label_debug_image_bytes = b""

        self.command_pub = rospy.Publisher(self.command_topic, String, queue_size=1, latch=True)
        self.home_pause_pub = rospy.Publisher(self.home_pause_topic, Bool, queue_size=1, latch=True)
        self.selected_label_pub = rospy.Publisher(self.selected_grasp_label_topic, String, queue_size=1, latch=True)
        self.carried_label_pub = rospy.Publisher(self.carried_grasp_label_topic, String, queue_size=1, latch=True)
        self.trial_context_pub = rospy.Publisher(self.trial_context_topic, String, queue_size=1, latch=True)
        self.study_event_pub = rospy.Publisher(self.study_event_topic, String, queue_size=20)
        self.ee_goal_pub = rospy.Publisher(self.ee_pose_goal_topic, EEPoseGoals, queue_size=1)
        self.relaxed_ik_reset_pub = rospy.Publisher(self.relaxed_ik_reset_topic, JointState, queue_size=1)
        self.manual_label_command_pub = rospy.Publisher(self.manual_label_command_topic, String, queue_size=10)

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=1)
        rospy.Subscriber(self.task_prompt_topic, String, self._task_prompt_cb, queue_size=1)
        rospy.Subscriber(self.execution_prompt_topic, String, self._execution_prompt_cb, queue_size=1)
        rospy.Subscriber(self.confirmation_prompt_topic, String, self._confirmation_prompt_cb, queue_size=1)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)
        rospy.Subscriber(self.study_event_topic, String, self._study_event_cb, queue_size=20)
        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=1)
        rospy.Subscriber(self.top_prob_topic, Float32, self._top_prob_cb, queue_size=1)
        rospy.Subscriber(self.selection_ready_topic, Bool, self._selection_ready_cb, queue_size=1)
        rospy.Subscriber(self.distribution_topic, Float32MultiArray, self._distribution_cb, queue_size=10)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_label_cb, queue_size=1)
        rospy.Subscriber(self.home_retreat_pose_topic, PoseStamped, self._home_retreat_pose_cb, queue_size=1)
        rospy.Subscriber(self.manual_label_state_topic, String, self._manual_label_state_cb, queue_size=1)
        rospy.Subscriber(self.manual_label_debug_image_topic, Image, self._manual_label_debug_image_cb, queue_size=1)

        self._publish_trial_context_locked()
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
        if isinstance(raw.get("tag_objects"), dict):
            entries = raw.get("tag_objects", {}) or {}
        elif isinstance(raw.get("candidate_objects"), dict):
            entries = raw.get("candidate_objects", {}) or {}
        else:
            entries = {}
        for key, meta in entries.items():
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
        if not self.enable_probability_logging:
            return ""
        os.makedirs(self.probability_log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.probability_log_dir, "user_study_probability_{}.jsonl".format(stamp))

    def _publish_study_event(self, event_name, **fields):
        target_source = self._step_source_info_locked()
        payload = {
            "event": str(event_name),
            "event_source": "dashboard",
            "stamp": rospy.Time.now().to_sec(),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "condition_id": self.condition_id,
            "block_id": self.block_id,
            "task_id": self.active_task_id or "",
            "step_id": self._current_step_locked().get("id", "") if self._current_step_locked() else "",
            "phase": self.current_phase or "",
            "target_source": target_source,
        }
        payload.update(fields)
        self.study_event_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _phase_cb(self, msg):
        with self.lock:
            self.current_phase = str(msg.data).strip()

    def _ee_cb(self, msg):
        with self.lock:
            self.latest_ee_pose = msg.pose

    def _home_retreat_pose_cb(self, msg):
        with self.lock:
            self.latest_home_retreat_pose = msg.pose

    def _task_prompt_cb(self, msg):
        with self.lock:
            self.task_prompt = str(msg.data).strip()

    def _execution_prompt_cb(self, msg):
        with self.lock:
            self.execution_prompt = str(msg.data).strip()

    def _confirmation_prompt_cb(self, msg):
        with self.lock:
            self.confirmation_prompt = str(msg.data).strip()

    def _top_goal_cb(self, msg):
        with self.lock:
            self.top_goal_label = str(msg.data).strip()

    def _top_prob_cb(self, msg):
        with self.lock:
            self.top_probability = float(msg.data)

    def _selection_ready_cb(self, msg):
        with self.lock:
            self.selection_ready = bool(msg.data)

    def _selected_label_cb(self, msg):
        with self.lock:
            self.selected_grasp_label = str(msg.data).strip()

    def _manual_label_state_cb(self, msg):
        try:
            payload = json.loads(str(msg.data))
        except Exception:
            payload = {"enabled": False}
        with self.lock:
            self.manual_label_state = payload

    def _manual_label_debug_image_cb(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return
        except Exception:
            return
        with self.lock:
            self.manual_label_debug_image_bytes = encoded.tobytes()

    def _publish_manual_label_command(self, action, **fields):
        payload = {"action": str(action)}
        payload.update(fields)
        self.manual_label_command_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _remove_candidate_everywhere_locked(self, tag_id):
        self._publish_command("remove_tag:{}".format(int(tag_id)))
        meta = self.tag_id_to_meta.get(int(tag_id), self.tag_id_to_meta.get(str(tag_id), {}))
        object_name = str(meta.get("object_name", "")).strip() if isinstance(meta, dict) else ""
        if object_name:
            self._publish_manual_label_command("remove", object_name=object_name)

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

    def _selected_grasp_meta_for_label(self, label):
        meta = self.label_to_meta.get(str(label).strip(), {})
        if not isinstance(meta, dict):
            return {}
        return meta

    def _append_probability_log(self, entry):
        if not self.enable_probability_logging or not self.probability_log_path:
            return
        with open(self.probability_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _step_source_info_locked(self, step=None):
        active_step = self._current_step_locked() if step is None else step
        if active_step is None:
            return "inactive"

        step_id = str(active_step.get("id", "")).strip().lower()
        if step_id == "select_lego_brick":
            return self.target_source_realtime_lego_label

        if step_id:
            return self.target_source_registry_label

        return self.target_source_hybrid_label

    def _distribution_cb(self, msg):
        with self.lock:
            probs = [float(v) for v in list(msg.data)]
            labels = self._distribution_labels_locked(len(probs))
            self.last_distribution = probs
            self.last_distribution_stamp = datetime.now().isoformat()
            task = self._current_task()
            step = self._current_step_locked()
            target_source = self._step_source_info_locked(step)
            entry = {
                "timestamp": self.last_distribution_stamp,
                "active_task_id": None if task is None else task["id"],
                "active_task_name": None if task is None else str(task.get("display_name", task["id"])),
                "active_step_index": self.active_step_index,
                "active_step_id": None if step is None else str(step.get("id", "")),
                "active_step_title": None if step is None else str(step.get("title", "")),
                "session_id": self.session_id,
                "participant_id": self.participant_id,
                "condition_id": self.condition_id,
                "block_id": self.block_id,
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
                "target_source": target_source,
            }
        try:
            self._append_probability_log(entry)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[user_study_dashboard] failed to append probability log: %s", exc)

    def _execution_state_cb(self, msg):
        text = str(msg.data).strip()
        with self.lock:
            previous = str(self.execution_state).strip().lower()
            self.execution_state = text
            current = text.lower()
            # If grasp confirmation is canceled and the executor returns to pregrasp selection,
            # clear the stale locked label so the UI falls back to the live intent target.
            if current == "wait_pregrasp_confirm" and previous == "wait_grasp_confirm":
                self.selected_grasp_label = ""

    def _study_event_cb(self, msg):
        try:
            event = json.loads(str(msg.data))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[user_study_dashboard] bad study event json: %s", exc)
            return

        event_name = str(event.get("event", "")).strip().lower()
        grasp_id = str(event.get("grasp_id", "")).strip()
        if not event_name:
            return

        with self.lock:
            if event_name == "confirm_cancel":
                self.selected_grasp_label = ""
                task = self._current_task()
                step = self._current_step_locked()
                stage = str(event.get("stage", "")).strip().lower()
                if (
                    task is not None
                    and step is not None
                    and str(task.get("id", "")).strip().lower() == "make_sandwich"
                    and str(step.get("id", "")).strip().lower() == "select_sandwich_item"
                    and stage in ("selection", "pregrasp", "grasp")
                ):
                    self.active_sandwich_item_label = ""
                    self._publish_carried_target_locked()
                return
            if not grasp_id:
                return
            task = self._current_task()
            if task is None or self.active_step_index is None:
                return
            step = task["steps"][self.active_step_index]
            if self._step_accepts_event(step, event_name, grasp_id):
                self._maybe_remove_completed_sandwich_object_locked(task, step, event_name, grasp_id)
                self._maybe_remove_completed_sort_object_locked(task, step, event_name, grasp_id)
                should_finish_task = self._should_finish_sandwich_task_locked(task, step, event_name)
                self._update_sandwich_item_tracking_locked(task, step, event_name, grasp_id)
                self._update_breakfast_item_tracking_locked(task, step, event_name, grasp_id)
                if should_finish_task:
                    self._finish_active_task_locked(task)
                    return
                self._advance_locked()

    def _publish_command(self, cmd):
        if not cmd:
            return
        self.command_pub.publish(String(data=str(cmd)))

    def _trial_context_payload_locked(self):
        task = self._current_task()
        step = self._current_step_locked()
        target_source = self._step_source_info_locked(step)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "active_task_id": None if task is None else task["id"],
            "active_task_name": None if task is None else str(task.get("display_name", task["id"])),
            "active_step_index": self.active_step_index,
            "active_step_id": None if step is None else str(step.get("id", "")),
            "active_step_title": None if step is None else str(step.get("title", "")),
            "active_step_description": None if step is None else str(step.get("description", "")),
            "active_step_manual": False if step is None else bool(step.get("manual", False)),
            "command": None if step is None else str(step.get("command", "")).strip(),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "condition_id": self.condition_id,
            "block_id": self.block_id,
            "completion_label": None if step is None else str(step.get("completion_label", "")).strip(),
            "success_event": None if step is None else str(step.get("success_event", "")).strip(),
            "completion_labels": [] if step is None else [
                str(item).strip()
                for item in list(step.get("completion_labels", []) or [])
                if str(item).strip()
            ],
            "completion_category": None if step is None else str(step.get("completion_category", "")).strip(),
            "completion_categories": [] if step is None else [
                str(item).strip()
                for item in list(step.get("completion_categories", []) or [])
                if str(item).strip()
            ],
            "allowed_tag_ids": [] if step is None else [
                int(item)
                for item in list(step.get("allowed_tag_ids", []) or [])
            ],
            "min_required_recorded_count": 0 if step is None else int(step.get("min_required_recorded_count", 0) or 0),
            "current_phase": self.current_phase,
            "target_source": target_source,
        }
        return payload

    def _publish_trial_context_locked(self):
        payload = self._trial_context_payload_locked()
        self.trial_context_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _clear_selected_target_locked(self):
        self.selected_grasp_label = ""
        self.selected_label_pub.publish(String(data=""))

    def _publish_carried_target_locked(self):
        self.carried_label_pub.publish(String(data=self.active_breakfast_item_label))

    def _current_task(self):
        if self.active_task_id is None:
            return None
        return self.tasks.get(self.active_task_id)

    def _activate_step_locked(self, step_index):
        task = self._current_task()
        if task is None:
            return
        self.active_step_index = step_index
        self.rescan_active = False
        self._clear_selected_target_locked()
        self._reset_intent_display_locked()
        self.execution_state = "idle"
        step = task["steps"][step_index]
        command = str(step.get("command", "")).strip()
        self._publish_trial_context_locked()
        if command:
            self._publish_command(command)

    def _reset_locked(self):
        self.active_task_id = None
        self.active_step_index = None
        self.rescan_active = False
        self.active_sandwich_item_label = ""
        self.active_breakfast_item_label = ""
        self._clear_selected_target_locked()
        self._publish_carried_target_locked()
        self.execution_state = "idle"
        self._reset_intent_display_locked()
        self._publish_trial_context_locked()
        self._publish_command(self.initial_reset_command)

    def start_task(self, task_id):
        with self.lock:
            if task_id not in self.tasks:
                raise KeyError(task_id)
            self._resume_after_home_locked()
            self.active_task_id = task_id
            self.active_step_index = None
            self.active_breakfast_item_label = ""
            self._publish_carried_target_locked()
            self._reset_intent_display_locked()
            task = self.tasks[task_id]
            self._publish_study_event("start_task", requested_task_id=task_id)
            if task["steps"]:
                self._activate_step_locked(0)

    def _reset_intent_display_locked(self):
        self.top_goal_label = ""
        self.top_probability = 0.0
        self.selection_ready = False
        self.last_distribution = []
        self.last_distribution_stamp = ""

    def reset_task(self):
        with self.lock:
            self._resume_after_home_locked()
            self._publish_study_event("reset_task")
            self._reset_locked()

    def scan_scene(self):
        with self.lock:
            self._resume_after_home_locked()
            self._publish_study_event("scan_scene")
            self.rescan_active = False
            self._clear_selected_target_locked()
            self.execution_state = "idle"
            self._reset_intent_display_locked()
            self._publish_trial_context_locked()
            self._publish_command(self.initial_reset_command)

    def manual_advance(self):
        with self.lock:
            task = self._current_task()
            if task is None or self.active_step_index is None:
                return
            step = task["steps"][self.active_step_index]
            if not bool(step.get("manual", False)):
                return
            self._publish_study_event("manual_advance")
            self._advance_locked()

    def quick_rescan_current_task(self):
        with self.lock:
            self._resume_after_home_locked()
            step = self._current_step_locked()
            if step is None:
                return
            self._publish_study_event("quick_rescan")
            self.rescan_active = True
            self._clear_selected_target_locked()
            self.execution_state = "idle"
            self._reset_intent_display_locked()
            self._publish_trial_context_locked()
            command = str(step.get("command", "")).strip()
            if command:
                self._publish_command(command)

    def manual_label_click(self, u, v):
        self._publish_manual_label_command("click", u=int(u), v=int(v))

    def manual_label_assign(self, object_name):
        self._publish_manual_label_command("assign", object_name=str(object_name).strip())

    def manual_label_remove(self, object_name):
        self._publish_manual_label_command("remove", object_name=str(object_name).strip())

    def manual_label_clear(self):
        self._publish_manual_label_command("clear")

    def send_robot_home(self):
        if Limb is None or RobotEnable is None or CHECK_VERSION is None:
            raise RuntimeError("intera_interface is not available in this environment")
        with self.lock:
            self._publish_study_event("send_home")
            self.active_task_id = None
            self.active_step_index = None
            self.rescan_active = False
            self.active_breakfast_item_label = ""
            self._clear_selected_target_locked()
            self._publish_carried_target_locked()
            self.execution_state = "idle"
            self._reset_intent_display_locked()
            self._publish_trial_context_locked()
        self._execute_home_motion()

    def _execute_home_motion(self):
        rospy.loginfo(
            "[user_study_dashboard] pausing teleop/controller and sending limb=%s to neutral",
            self.home_limb,
        )
        self.home_pause_pub.publish(Bool(data=True))
        with self.lock:
            self.home_hold_active = True
        rospy.sleep(0.3)
        try:
            rs = RobotEnable(CHECK_VERSION)
            rs.enable()
            limb = Limb(self.home_limb)
            self._move_to_home_retreat_pose()
            ok = limb.move_to_neutral(timeout=self.home_timeout, speed=self.home_speed)
            if not ok:
                raise RuntimeError("move_to_neutral returned failure")
            self._publish_relaxed_ik_reset(limb)
            self._publish_current_pose_hold_goal(limb)
        finally:
            rospy.sleep(0.8)
            with self.lock:
                self._resume_after_home_locked()

    def _resume_after_home_locked(self):
        if not self.home_hold_active:
            return
        self.home_hold_active = False
        self.home_pause_pub.publish(Bool(data=False))

    def _current_limb_pose(self, limb):
        try:
            endpoint = limb.endpoint_pose()
        except Exception as exc:
            rospy.logwarn("[user_study_dashboard] failed to query endpoint pose from limb: %s", exc)
            return None

        if not isinstance(endpoint, dict):
            return None
        position = endpoint.get("position")
        orientation = endpoint.get("orientation")
        if position is None or orientation is None:
            return None

        pose = Pose()
        pose.position.x = float(position.x)
        pose.position.y = float(position.y)
        pose.position.z = float(position.z)
        pose.orientation.x = float(orientation.x)
        pose.orientation.y = float(orientation.y)
        pose.orientation.z = float(orientation.z)
        pose.orientation.w = float(orientation.w)
        return pose

    @staticmethod
    def _pose_position_error(pose_a, pose_b):
        if pose_a is None or pose_b is None:
            return None
        dx = float(pose_a.position.x) - float(pose_b.position.x)
        dy = float(pose_a.position.y) - float(pose_b.position.y)
        dz = float(pose_a.position.z) - float(pose_b.position.z)
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _move_to_home_retreat_pose(self):
        with self.lock:
            retreat_pose = None
            if self.latest_home_retreat_pose is not None:
                retreat_pose = Pose()
                retreat_pose.position = self.latest_home_retreat_pose.position
                retreat_pose.orientation = self.latest_home_retreat_pose.orientation
        if retreat_pose is None:
            rospy.loginfo("[user_study_dashboard] no retreat pose available before home; skipping pre-home retreat")
            return

        rospy.loginfo("[user_study_dashboard] moving to retreat/pregrasp pose before home")
        deadline = rospy.Time.now() + rospy.Duration(max(0.1, self.home_retreat_timeout))
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            msg = EEPoseGoals()
            msg.header.stamp = rospy.Time.now()
            msg.ee_poses.append(retreat_pose)
            msg.tolerances.append(Twist())
            self.ee_goal_pub.publish(msg)
            with self.lock:
                current_pose = self.latest_ee_pose
            error_m = self._pose_position_error(current_pose, retreat_pose)
            if error_m is not None and error_m <= self.home_retreat_position_tolerance:
                rospy.loginfo("[user_study_dashboard] reached retreat pose before home")
                return
            rate.sleep()
        rospy.loginfo("[user_study_dashboard] retreat pose timeout expired; continuing to home")

    def _publish_current_pose_hold_goal(self, limb=None):
        pose = self._current_limb_pose(limb) if limb is not None else None
        with self.lock:
            if pose is None and self.latest_ee_pose is not None:
                pose = self.latest_ee_pose
        if pose is None:
            rospy.logwarn("[user_study_dashboard] no current ee pose available to overwrite RelaxedIK goal after home")
            return

        rospy.loginfo("[user_study_dashboard] publishing actual current pose to overwrite stale RelaxedIK target")
        rate = rospy.Rate(20)
        for _ in range(10):
            msg = EEPoseGoals()
            msg.header.stamp = rospy.Time.now()
            msg.ee_poses.append(pose)
            msg.tolerances.append(Twist())
            self.ee_goal_pub.publish(msg)
            rate.sleep()

    def _publish_relaxed_ik_reset(self, limb):
        try:
            joint_names = list(limb.joint_names())
            joint_angles = limb.joint_angles()
        except Exception as exc:
            rospy.logwarn("[user_study_dashboard] failed to query joint angles for RelaxedIK reset: %s", exc)
            return

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        msg.position = [float(joint_angles[name]) for name in joint_names]
        rospy.loginfo("[user_study_dashboard] publishing RelaxedIK reset at current neutral joint state")
        for _ in range(3):
            msg.header.stamp = rospy.Time.now()
            self.relaxed_ik_reset_pub.publish(msg)
            rospy.sleep(0.05)

    def _advance_locked(self):
        task = self._current_task()
        if task is None or self.active_step_index is None:
            return
        next_index = self.active_step_index + 1
        if next_index >= len(task["steps"]):
            if bool(task.get("repeat_after_completion", False)):
                repeat_from_index = int(task.get("repeat_from_step_index", 0) or 0)
                repeat_from_index = max(0, min(repeat_from_index, len(task["steps"]) - 1))
                self._activate_step_locked(repeat_from_index)
                return
            self._finish_active_task_locked(task)
            return
        self._activate_step_locked(next_index)

    def _finish_active_task_locked(self, task):
        self.active_step_index = None
        self.active_task_id = None
        self.active_sandwich_item_label = ""
        self.active_breakfast_item_label = ""
        self.breakfast_pour_active = False
        self._publish_carried_target_locked()
        self._publish_trial_context_locked()
        self._publish_command(str(task.get("completion_reset_command", self.initial_reset_command)).strip())

    def _maybe_remove_completed_sort_object_locked(self, task, step, event_name, grasp_id):
        task_id = str(task.get("id", "")).strip().lower()
        step_id = str(step.get("id", "")).strip().lower()
        if task_id != "sorting" or step_id != "select_sort_object" or event_name != "grasp_complete":
            return
        meta = self.label_to_meta.get(grasp_id, {})
        if str(meta.get("category", "")).strip() == "destination":
            return
        tag_id = self._tag_id_for_completion_label(grasp_id)
        if tag_id is None:
            rospy.logwarn(
                "[user_study_dashboard] could not resolve sorting grasp label %s to a tag id for registry removal",
                grasp_id,
            )
            return
        self._remove_candidate_everywhere_locked(tag_id)

    def _update_sandwich_item_tracking_locked(self, task, step, event_name, grasp_id):
        task_id = str(task.get("id", "")).strip().lower()
        step_id = str(step.get("id", "")).strip().lower()
        if task_id != "make_sandwich":
            return
        if step_id == "select_sandwich_item" and event_name == "grasp_complete":
            self.active_sandwich_item_label = str(grasp_id).strip()
            return
        if step_id == "place_sandwich_item" and event_name == "release_complete":
            self.active_sandwich_item_label = ""

    def _update_breakfast_item_tracking_locked(self, task, step, event_name, grasp_id):
        task_id = str(task.get("id", "")).strip().lower()
        if task_id != "make_breakfast":
            return
        step_id = str(step.get("id", "")).strip().lower()
        if step_id in ("select_breakfast_ingredient", "select_breakfast_milk") and event_name == "grasp_complete":
            self.active_breakfast_item_label = str(grasp_id).strip()
            self.breakfast_pour_active = False
            self._publish_carried_target_locked()
            return
        if step_id in ("pour_breakfast_ingredient", "pour_breakfast_milk") and event_name == "pour_start":
            self.breakfast_pour_active = True
            if grasp_id:
                self.active_breakfast_item_label = str(grasp_id).strip()
            self._publish_carried_target_locked()
            return
        if step_id in ("pour_breakfast_ingredient", "pour_breakfast_milk") and event_name == "pour_complete":
            self.breakfast_pour_active = False
            self.active_breakfast_item_label = ""
            self._publish_carried_target_locked()

    def _should_finish_sandwich_task_locked(self, task, step, event_name):
        task_id = str(task.get("id", "")).strip().lower()
        step_id = str(step.get("id", "")).strip().lower()
        if task_id != "make_sandwich":
            return False
        if step_id != "place_sandwich_item" or event_name != "release_complete":
            return False
        return str(self.active_sandwich_item_label).strip() == "bread_top_grasp"

    def _maybe_remove_completed_sandwich_object_locked(self, task, step, event_name, grasp_id):
        task_id = str(task.get("id", "")).strip().lower()
        step_id = str(step.get("id", "")).strip().lower()
        if task_id != "make_sandwich":
            return
        carried_label = ""
        if step_id == "select_sandwich_item" and event_name == "grasp_complete":
            carried_label = str(grasp_id).strip()
        elif step_id == "place_sandwich_item" and event_name == "release_complete":
            carried_label = str(grasp_id).strip() or str(self.active_sandwich_item_label).strip()
        else:
            return
        if not carried_label:
            return
        tag_id = self._tag_id_for_completion_label(carried_label)
        if tag_id is None:
            rospy.logwarn(
                "[user_study_dashboard] could not resolve sandwich grasp label %s to a tag id for registry removal",
                carried_label,
            )
            return
        self._remove_candidate_everywhere_locked(tag_id)

    def _tag_id_for_completion_label(self, label):
        label = str(label).strip()
        if not label:
            return None
        for tag_id, meta in self.tag_id_to_meta.items():
            if str(meta.get("grasp_complete_label", "")).strip() == label:
                try:
                    return int(tag_id)
                except Exception:
                    return None
        return None

    def _step_matches_label(self, step, label):
        meta = self.label_to_meta.get(label, {})
        completion_label = str(step.get("completion_label", "")).strip()
        completion_labels = [
            str(item).strip()
            for item in list(step.get("completion_labels", []) or [])
            if str(item).strip()
        ]
        completion_category = str(step.get("completion_category", "")).strip()
        completion_categories = [
            str(item).strip()
            for item in list(step.get("completion_categories", []) or [])
            if str(item).strip()
        ]
        if completion_label and completion_label == label:
            return True
        if completion_labels and label in completion_labels:
            return True
        if completion_category and str(meta.get("category", "")).strip() == completion_category:
            return True
        if completion_categories and str(meta.get("category", "")).strip() in completion_categories:
            return True
        return False

    def _step_expects_release(self, step, label):
        success_event = str(step.get("success_event", "")).strip().lower()
        if success_event:
            return success_event == "release_complete"
        step_id = str(step.get("id", "")).strip().lower()
        if "destination" in step_id:
            return True
        category = str(self.label_to_meta.get(label, {}).get("category", "")).strip()
        return category == "destination"

    def _step_accepts_event(self, step, event_name, label):
        success_event = str(step.get("success_event", "")).strip().lower()
        if success_event:
            return self._step_matches_label(step, label) and event_name == success_event
        if not self._step_matches_label(step, label):
            return False
        expected_event = "release_complete" if self._step_expects_release(step, label) else "grasp_complete"
        return event_name == expected_event

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

    def _current_step_categories_locked(self):
        step = self._current_step_locked()
        if step is None:
            return []
        allowed_tag_ids = [
            int(item)
            for item in list(step.get("allowed_tag_ids", []) or [])
        ]
        if allowed_tag_ids:
            categories = []
            for tag_id in allowed_tag_ids:
                meta = self.tag_id_to_meta.get(str(tag_id), {})
                category = str(meta.get("category", "")).strip()
                if category and category not in categories:
                    categories.append(category)
            if categories:
                return categories
        categories = []
        completion_category = str(step.get("completion_category", "")).strip()
        if completion_category:
            categories.append(completion_category)
        for item in list(step.get("completion_categories", []) or []):
            value = str(item).strip()
            if value and value not in categories:
                categories.append(value)
        if categories:
            return categories
        command = str(step.get("command", "")).strip().lower()
        if command in ("grasp_milk", "select_milk", "milk"):
            return ["milk"]
        if command in ("grasp_condiment", "select_condiment", "condiment"):
            return ["cereal", "chocolate"]
        if command in ("grasp_cereal", "select_cereal", "cereal"):
            return ["cereal"]
        if command in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return ["chocolate"]
        return []

    def _object_view_for_tag_id(self, tag_id):
        meta = self.tag_id_to_meta.get(str(tag_id), {})
        object_name = str(meta.get("object_name", "tag_{}".format(tag_id))).strip()
        category = str(meta.get("category", "")).strip()
        return {
            "tag_id": int(tag_id),
            "label": str(tag_id),
            "object_name": object_name or "tag_{}".format(tag_id),
            "category": category,
        }

    def _manual_labeled_candidate_ids_locked(self):
        payload = self.manual_label_state if isinstance(self.manual_label_state, dict) else {}
        labeled = payload.get("labeled_objects", []) if isinstance(payload, dict) else []
        candidate_ids = set()
        for item in list(labeled or []):
            if not isinstance(item, dict):
                continue
            try:
                candidate_ids.add(int(item.get("candidate_id")))
            except Exception:
                continue
        return candidate_ids

    def _scene_readiness_locked(self):
        categories = self._current_step_categories_locked()
        step = self._current_step_locked()
        step_id = "" if step is None else str(step.get("id", "")).strip().lower()
        if not categories:
            return {
                "status": "idle",
                "title": "Task Status",
                "message": "Wait for the next step, or scan the workspace if the operator is preparing the task.",
                "scope_label": "No task selected. Global scene scanning is available.",
                "required_count": 0,
                "recorded_count": 0,
                "allowed_objects": [],
                "allowed_note": "Only objects shown under Recorded now are currently graspable.",
                "recorded_objects": [],
                "missing_objects": [],
                "rescan_active": False,
            }

        if step_id in ("pour_breakfast_ingredient", "pour_breakfast_milk"):
            carried_label = str(self.active_breakfast_item_label).strip()
            if self.breakfast_pour_active or carried_label:
                return {
                    "status": "ready",
                    "title": "Task Status",
                    "message": "Automatic pour step is ready. The robot is using the currently grasped breakfast item and does not require a new scene scan.",
                    "scope_label": "Current task categories: {}".format(", ".join(categories)),
                    "required_count": 1,
                    "recorded_count": 1,
                    "allowed_objects": [],
                    "allowed_note": "The currently grasped breakfast item will be used for the automatic pour sequence.",
                    "recorded_objects": [],
                    "missing_objects": [],
                    "rescan_active": False,
                }

        required_ids = [
            int(item)
            for item in list(step.get("allowed_tag_ids", []) or [])
        ] if step is not None else []
        if not required_ids:
            required_ids = []
            for tag_id, meta in self.tag_id_to_meta.items():
                category = str(meta.get("category", "")).strip()
                if category in categories:
                    required_ids.append(int(tag_id))
        required_ids = sorted(set(required_ids))

        recorded_ids = set()
        for label in self.latest_candidate_labels:
            try:
                recorded_ids.add(int(label))
            except Exception:
                continue
        if step_id in ("select_sandwich_item", "place_sandwich_item"):
            recorded_ids.update(self._manual_labeled_candidate_ids_locked())

        recorded_task_ids = [tag_id for tag_id in required_ids if tag_id in recorded_ids]
        missing_ids = [tag_id for tag_id in required_ids if tag_id not in recorded_ids]
        min_required_count = 0 if step is None else int(step.get("min_required_recorded_count", 0) or 0)
        if min_required_count <= 0:
            min_required_count = len(required_ids)

        if len(recorded_task_ids) < min_required_count:
            status = "needs_rescan"
            if min_required_count < len(required_ids):
                message = (
                    "Recorded scene has {} of {} available task objects. "
                    "At least {} are required for this step. Use Quick Rescan Current Task if you want more destination choices."
                ).format(len(recorded_task_ids), len(required_ids), min_required_count)
            else:
                message = (
                    "Recorded scene is missing {} of {} required objects. "
                    "Use Quick Rescan Current Task, then point the wrist camera only at the missing task objects."
                ).format(len(missing_ids), len(required_ids))
        else:
            status = "ready"
            if min_required_count < len(required_ids):
                message = (
                    "Recorded scene has {} available task objects, which meets the minimum requirement of {} for this step."
                ).format(len(recorded_task_ids), min_required_count)
            else:
                message = (
                    "Recorded scene already contains all {} required task objects. "
                    "You can start directly without scanning the full workspace."
                ).format(len(required_ids))

        return {
            "status": status,
            "title": "Task Status",
            "message": message,
            "scope_label": "Current task categories: {}".format(", ".join(categories)),
            "required_count": min_required_count,
            "recorded_count": len(recorded_task_ids),
            "allowed_objects": [self._object_view_for_tag_id(tag_id) for tag_id in required_ids],
            "allowed_note": "Allowed this step lists all valid task objects. Only objects shown under Recorded now are currently graspable.",
            "recorded_objects": [self._object_view_for_tag_id(tag_id) for tag_id in recorded_task_ids],
            "missing_objects": [self._object_view_for_tag_id(tag_id) for tag_id in missing_ids],
            "rescan_active": bool(self.rescan_active and len(recorded_task_ids) < min_required_count),
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
                "confirmation_prompt": self.confirmation_prompt,
                "execution_state": self.execution_state,
                "top_goal_label": self.top_goal_label,
                "top_goal_object_name": str(self._tag_meta_for_label(self.top_goal_label).get("object_name", "")),
                "top_probability": self.top_probability,
                "selection_ready": bool(self.selection_ready),
                "participant_takeover_threshold": self.participant_takeover_threshold,
                "selected_grasp_label": self.selected_grasp_label,
                "selected_grasp_object_name": str(
                    self._selected_grasp_meta_for_label(self.selected_grasp_label).get("object_name", "")
                ),
                "active_breakfast_item_label": self.active_breakfast_item_label,
                "breakfast_pour_active": bool(self.breakfast_pour_active),
                "active_breakfast_item_object_name": str(
                    self._selected_grasp_meta_for_label(self.active_breakfast_item_label).get("object_name", "")
                ),
                "probability_log_path": self.probability_log_path,
                "scene_readiness": self._scene_readiness_locked(),
                "manual_labeler": dict(self.manual_label_state),
            }

    def _make_server(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html", "/participant", "/participant.html"):
                    body = _render_dashboard_html("participant").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path in ("/operator", "/operator.html"):
                    body = _render_dashboard_html("operator").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/static/participant_controller_image":
                    image_path = dashboard.participant_controller_image
                    if not os.path.exists(image_path):
                        self.send_error(404)
                        return
                    with open(image_path, "rb") as handle:
                        body = handle.read()
                    lower_path = image_path.lower()
                    if lower_path.endswith(".webp"):
                        content_type = "image/webp"
                    elif lower_path.endswith(".png"):
                        content_type = "image/png"
                    else:
                        content_type = "image/jpeg"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/state" or self.path.startswith("/api/state?"):
                    body = json.dumps(dashboard.state_payload()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/api/manual_label_image"):
                    with dashboard.lock:
                        body = bytes(dashboard.manual_label_debug_image_bytes)
                    if not body:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
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
                    elif self.path == "/api/scan_scene":
                        dashboard.scan_scene()
                    elif self.path == "/api/manual_advance":
                        dashboard.manual_advance()
                    elif self.path == "/api/quick_rescan":
                        dashboard.quick_rescan_current_task()
                    elif self.path == "/api/send_home":
                        dashboard.send_robot_home()
                    elif self.path == "/api/manual_label_click":
                        dashboard.manual_label_click(int(payload.get("u", 0)), int(payload.get("v", 0)))
                    elif self.path == "/api/manual_label_assign":
                        dashboard.manual_label_assign(str(payload.get("object_name", "")).strip())
                    elif self.path == "/api/manual_label_remove":
                        dashboard.manual_label_remove(str(payload.get("object_name", "")).strip())
                    elif self.path == "/api/manual_label_clear":
                        dashboard.manual_label_clear()
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
                except RuntimeError as exc:
                    body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                    self.send_response(500)
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

        server = ReusableThreadingHTTPServer((self.host, self.port), Handler)
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
    try:
        UserStudyDashboard().run()
    except Exception:
        rospy.logerr("[user_study_dashboard] fatal startup error:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
