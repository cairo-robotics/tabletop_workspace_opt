#!/usr/bin/env python3
"""Small web dashboard for the CASPER-lite decision-making pipeline."""

import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import rospy
import yaml
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray, String


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CASPER-Lite Decision Pipeline</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #181a20;
      --panel2: #20232b;
      --text: #f4f6fb;
      --muted: #a7adbb;
      --line: #343947;
      --accent: #38bdf8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.15fr) minmax(360px, .85fr);
      gap: 14px;
      padding: 14px;
      min-height: 100vh;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    h1 { font-size: 22px; margin: 0 0 4px; }
    h2 { font-size: 15px; margin: 0 0 10px; color: var(--muted); font-weight: 650; }
    .sub { color: var(--muted); font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
    .stat { background: var(--panel2); border: 1px solid var(--line); border-radius: 7px; padding: 10px; min-height: 70px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { margin-top: 5px; font-size: 22px; font-weight: 750; overflow-wrap: anywhere; }
    .small-value { font-size: 16px; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 999px; border: 1px solid var(--line); background: var(--panel2); font-size: 13px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
    .dot.good { background: var(--good); }
    .dot.bad { background: var(--bad); }
    .prompt { margin-top: 14px; background: #0f1720; border: 1px solid var(--line); border-radius: 7px; padding: 12px; font-size: 18px; line-height: 1.35; min-height: 72px; }
    .bars { display: flex; flex-direction: column; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 150px 1fr 54px; gap: 10px; align-items: center; }
    .bar-name { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 14px; }
    .track { height: 24px; background: #0d1016; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: var(--accent); transition: width .18s ease; }
    .score { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
    .image-stack { display: flex; flex-direction: column; gap: 14px; }
    img { width: 100%; max-height: calc((100vh - 150px) / 2); object-fit: contain; background: #050608; border: 1px solid var(--line); border-radius: 8px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #0b0d12; border: 1px solid var(--line); border-radius: 7px; padding: 10px; color: #d8dee9; font-size: 12px; max-height: 210px; overflow: auto; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <section>
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
      <div>
        <h1>CASPER-Lite Decision Pipeline</h1>
        <div class="sub">VLM intent inference, self-consistency confidence, and user-confirmed takeover</div>
      </div>
      <div class="pill"><span id="live-dot" class="dot"></span><span id="live-text">waiting</span></div>
    </div>
    <div class="grid">
      <div class="stat"><div class="label">VLM Intent</div><div id="vlm-intent" class="value">-</div></div>
      <div class="stat"><div class="label">Intent Output</div><div id="top-goal" class="value">-</div></div>
      <div class="stat"><div class="label">Confidence</div><div id="confidence" class="value">0.00</div></div>
      <div class="stat"><div class="label">Latency</div><div id="latency" class="value">-</div></div>
    </div>
    <div class="prompt" id="prompt">Waiting for CASPER prediction...</div>
    <section style="margin-top:14px;padding:12px;">
      <h2>Intent Confidence Bars</h2>
      <div id="bars" class="bars"></div>
    </section>
    <section style="margin-top:14px;padding:12px;">
      <h2>Raw VLM Decision</h2>
      <pre id="raw">{}</pre>
    </section>
  </section>
  <section class="image-stack">
    <div>
      <h2>Wrist Visual Prompt</h2>
      <img id="frame" alt="CASPER wrist visual prompt">
    </div>
    <div>
      <h2>Top-Down Semantic Map</h2>
      <img id="semantic-frame" alt="CASPER top-down semantic map">
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
function fmt(v, n=2) {
  const x = Number(v);
  return Number.isFinite(x) ? x.toFixed(n) : "-";
}
function objectLabel(item) {
  const name = item.object_name || item.label || "";
  return item.label ? `${item.label} ${name}` : name || "-";
}
function renderBars(items) {
  const bars = $("bars");
  bars.innerHTML = "";
  if (!items.length) {
    bars.innerHTML = '<div class="sub">No distribution published yet.</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("div");
    name.className = "bar-name";
    name.textContent = objectLabel(item);
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${Math.max(0, Math.min(1, item.probability || 0)) * 100}%`;
    if (String(item.label) === String(window.latestTopGoal || "")) fill.style.background = "var(--good)";
    track.appendChild(fill);
    const score = document.createElement("div");
    score.className = "score";
    score.textContent = fmt(item.probability || 0);
    row.appendChild(name);
    row.appendChild(track);
    row.appendChild(score);
    bars.appendChild(row);
  }
}
async function refresh() {
  const res = await fetch("/state", {cache: "no-store"});
  const data = await res.json();
  window.latestTopGoal = data.top_goal_label;
  const age = data.casper_prediction_age_sec;
  const live = Number.isFinite(age) && age < 15;
  $("live-dot").className = "dot " + (live ? "good" : "bad");
  $("live-text").textContent = live ? "live" : "stale";
  $("vlm-intent").textContent = data.casper_object_name || data.casper_predicted_candidate_id || "-";
  $("top-goal").textContent = data.top_goal_object_name || data.top_goal_label || "-";
  $("confidence").textContent = fmt(data.top_probability);
  $("latency").textContent = Number.isFinite(Number(data.casper_latency_sec_total)) ? `${fmt(data.casper_latency_sec_total, 1)}s` : "-";
  $("prompt").textContent = data.confirmation_prompt || data.intent_status || data.casper_status || "Waiting for CASPER prediction...";
  renderBars(data.distribution || []);
  $("raw").textContent = JSON.stringify(data.casper_prediction || {}, null, 2);
  if (data.casper_image_url) {
    $("frame").src = data.casper_image_url + "&t=" + Date.now();
  }
  if (data.casper_semantic_map_url) {
    $("semantic-frame").src = data.casper_semantic_map_url + "&t=" + Date.now();
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""


class CasperLiteDecisionDashboard(object):
    def __init__(self):
        rospy.init_node("casper_lite_decision_dashboard")
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.allowed_file_root = os.path.abspath(rospy.get_param("~allowed_file_root", package_root))
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", os.path.join(package_root, "config", "apriltag_object_map.yaml"))
        )
        self.host = str(rospy.get_param("~host", "0.0.0.0")).strip()
        self.port = int(rospy.get_param("~port", 8770))
        self.casper_prediction_topic = str(rospy.get_param("~casper_prediction_topic", "/casper_lite/prediction")).strip()
        self.casper_status_topic = str(rospy.get_param("~casper_status_topic", "/casper_lite/status")).strip()
        self.intent_status_topic = str(rospy.get_param("~intent_status_topic", "/apriltag_intent_inference/status")).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_probability_topic = str(rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")).strip()
        self.distribution_topic = str(rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")).strip()
        self.distribution_labels_topic = str(
            rospy.get_param("~distribution_labels_topic", "/apriltag_intent_inference/distribution_labels")
        ).strip()
        self.confirmation_prompt_topic = str(
            rospy.get_param("~confirmation_prompt_topic", "/apriltag_intent_inference/confirmation_prompt")
        ).strip()
        self.selection_ready_topic = str(rospy.get_param("~selection_ready_topic", "/intent_inference/selection_ready")).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()

        self.lock = threading.RLock()
        self.tag_meta = self._load_object_map()
        self.state = {
            "casper_prediction": {},
            "casper_prediction_stamp": 0.0,
            "casper_status": "",
            "intent_status": "",
            "top_goal_label": "",
            "top_probability": 0.0,
            "distribution": [],
            "distribution_labels": [],
            "confirmation_prompt": "",
            "selection_ready": False,
            "execution_state": "",
        }

        rospy.Subscriber(self.casper_prediction_topic, String, self._casper_prediction_cb, queue_size=10)
        rospy.Subscriber(self.casper_status_topic, String, self._casper_status_cb, queue_size=10)
        rospy.Subscriber(self.intent_status_topic, String, self._intent_status_cb, queue_size=10)
        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=10)
        rospy.Subscriber(self.top_probability_topic, Float32, self._top_probability_cb, queue_size=10)
        rospy.Subscriber(self.distribution_topic, Float32MultiArray, self._distribution_cb, queue_size=10)
        rospy.Subscriber(self.distribution_labels_topic, Int32MultiArray, self._distribution_labels_cb, queue_size=10)
        rospy.Subscriber(self.confirmation_prompt_topic, String, self._confirmation_prompt_cb, queue_size=10)
        rospy.Subscriber(self.selection_ready_topic, Bool, self._selection_ready_cb, queue_size=10)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)

        self.httpd = self._make_server()
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("[casper_lite_decision_dashboard] ready at http://%s:%d", self.host, self.port)

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("tag_objects") or raw.get("candidate_objects") or {}
        return {str(k): dict(v or {}) for k, v in entries.items() if isinstance(v, dict)}

    def _object_name(self, label):
        meta = self.tag_meta.get(str(label), {})
        return str(meta.get("object_name") or meta.get("grasp_complete_label") or "").strip()

    def _casper_prediction_cb(self, msg):
        try:
            payload = json.loads(str(msg.data))
        except Exception:
            return
        with self.lock:
            self.state["casper_prediction"] = payload
            self.state["casper_prediction_stamp"] = time.time()

    def _casper_status_cb(self, msg):
        with self.lock:
            self.state["casper_status"] = str(msg.data)

    def _intent_status_cb(self, msg):
        with self.lock:
            self.state["intent_status"] = str(msg.data)

    def _top_goal_cb(self, msg):
        with self.lock:
            self.state["top_goal_label"] = str(msg.data).strip()

    def _top_probability_cb(self, msg):
        with self.lock:
            self.state["top_probability"] = float(msg.data)

    def _distribution_cb(self, msg):
        with self.lock:
            self.state["distribution"] = [float(v) for v in list(msg.data)]

    def _distribution_labels_cb(self, msg):
        with self.lock:
            self.state["distribution_labels"] = [str(int(v)) for v in list(msg.data)]

    def _confirmation_prompt_cb(self, msg):
        with self.lock:
            self.state["confirmation_prompt"] = str(msg.data)

    def _selection_ready_cb(self, msg):
        with self.lock:
            self.state["selection_ready"] = bool(msg.data)

    def _execution_state_cb(self, msg):
        with self.lock:
            self.state["execution_state"] = str(msg.data)

    def _state_payload(self):
        with self.lock:
            state = dict(self.state)
            prediction = dict(state.get("casper_prediction") or {})
        labels = list(state.get("distribution_labels") or [])
        probs = list(state.get("distribution") or [])
        bars = []
        for idx, prob in enumerate(probs):
            label = labels[idx] if idx < len(labels) else str(idx)
            bars.append(
                {
                    "label": label,
                    "object_name": self._object_name(label),
                    "probability": float(prob),
                }
            )
        casper_id = str(prediction.get("predicted_candidate_id") or "")
        image_path = str(prediction.get("image_path") or "")
        semantic_map_path = str(prediction.get("semantic_map_path") or "")
        state.update(
            {
                "distribution": bars,
                "top_goal_object_name": self._object_name(state.get("top_goal_label", "")),
                "casper_predicted_candidate_id": casper_id,
                "casper_object_name": self._object_name(casper_id),
                "casper_latency_sec_total": prediction.get("latency_sec_total"),
                "casper_prediction_age_sec": time.time() - float(state.get("casper_prediction_stamp") or 0.0),
                "casper_image_url": "/image?path={}".format(quote(image_path)) if image_path else "",
                "casper_semantic_map_url": "/image?path={}".format(quote(semantic_map_path)) if semantic_map_path else "",
            }
        )
        return state

    def _file_allowed(self, path):
        real = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        root = os.path.realpath(self.allowed_file_root)
        return real == root or real.startswith(root + os.sep)

    def _make_server(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt, *_args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    body = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/state":
                    body = json.dumps(dashboard._state_payload(), sort_keys=True).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/image":
                    query = parse_qs(parsed.query)
                    path = query.get("path", [""])[0]
                    path = os.path.abspath(os.path.expanduser(path))
                    if not path or not os.path.exists(path) or not dashboard._file_allowed(path):
                        self.send_error(404)
                        return
                    with open(path, "rb") as handle:
                        body = handle.read()
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

        return ThreadingHTTPServer((self.host, self.port), Handler)

    def _shutdown(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    CasperLiteDecisionDashboard()
    rospy.spin()
