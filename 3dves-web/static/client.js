/**
 * client.js
 * =========
 * WebSocket lifecycle, view state, input handling, and inspection panel
 * for the 3DVES live simulation browser client.
 *
 * Imports drawFrame, worldToCanvas, canvasToWorld, toggleC4IView, setRoadData
 * from renderer.js.  All canvas drawing is delegated there; this module
 * owns only data flow and user interaction.
 *
 * View state
 * ----------
 * viewState = { offsetX, offsetY, zoom, worldX, worldY }
 *
 * offsetX/Y : world-metre coordinates of the canvas bottom-left corner.
 * zoom      : scale multiplier (1.0 = world fills the shorter canvas dim).
 *
 * Zoom anchors to the cursor position: the world point under the cursor
 * stays fixed in canvas pixels after the zoom update.
 */

import {
  drawFrame,
  worldToCanvas,
  canvasToWorld,
  toggleC4IView,
  setRoadData,
} from "./renderer.js";

// ── State ─────────────────────────────────────────────────────────────────
const canvas = document.getElementById("sim-canvas");

let ws          = null;
let scenarioId  = null;
let lastFrame   = null;
let paused      = false;
let selectedId  = null;
let c4iActive   = true;

// View state — populated from the server "hello" frame
const vs = {
  worldX: 1500, worldY: 1500,
  offsetX: 0, offsetY: 0,
  zoom: 1.0,
};

// Display filters — kept in sync with checkbox DOM elements
const opts = {
  uav: true, wheeled: true, tracked: true, ped: true,
  tracks: true, fov: true, roads: true,
};

// Right-drag pan
let _dragging    = false;
let _dragStartX  = 0;
let _dragStartY  = 0;
let _dragOffX    = 0;
let _dragOffY    = 0;

// ── Resize canvas to CSS size ─────────────────────────────────────────────
function _resize() {
  canvas.width  = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
}
window.addEventListener("resize", _resize);
_resize();

// ── View helpers ──────────────────────────────────────────────────────────

function resetView() {
  vs.zoom    = 1.0;
  vs.offsetX = 0;
  vs.offsetY = 0;
}

function _pixelsPerMetre() {
  return Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
}

// ── WebSocket ─────────────────────────────────────────────────────────────

function connect(sid) {
  if (ws) { ws.close(); ws = null; }
  scenarioId = sid;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/sim/${sid}`);

  ws.onopen = () => {
    _setStatus("Connected", true);
    document.getElementById("btn-pause").disabled = false;
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);

    // First message is the "hello" frame with world dimensions
    if (data.type === "hello") {
      vs.worldX = data.world_x;
      vs.worldY = data.world_y;
      resetView();
      if (data.road_data) setRoadData(data.road_data);
      return;
    }

    lastFrame = data;
    drawFrame(canvas, lastFrame, vs, opts, selectedId);
  };

  ws.onclose = () => {
    _setStatus("Disconnected", false);
    document.getElementById("btn-pause").disabled = true;
    // Attempt reconnect after 2 s if the scenario is still alive
    setTimeout(() => {
      if (scenarioId) {
        fetch(`/scenarios/${scenarioId}/info`)
          .then(r => r.ok ? r.json() : null)
          .then(info => { if (info && info.alive) connect(scenarioId); })
          .catch(() => {});
      }
    }, 2000);
  };

  ws.onerror = () => _setStatus("Error", false);
}

function sendCommand(action, extra = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action, ...extra }));
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────

function _setStatus(text, connected) {
  document.getElementById("hud-status").textContent = text;
  const dot = document.getElementById("conn-dot");
  dot.classList.toggle("connected", connected);
}

function _showPanel(ent) {
  selectedId = ent.id;
  document.getElementById("panel-title").textContent = ent.type + " · " + ent.id.slice(0, 8);
  const body = document.getElementById("panel-body");

  const row = (k, v, cls = "") =>
    `<div class="prow"><span class="pk">${k}</span><span class="pv ${cls}">${v}</span></div>`;

  let html = "";
  html += row("State",   ent.state);
  html += row("EOI",     ent.is_eoi  ? "YES" : "No",   ent.is_eoi  ? "eoi" : "");
  html += row("Detected",ent.detected ? "YES" : "No",   ent.detected ? "det" : "");

  const p = ent.pos;
  html += row("Position", `${p[0].toFixed(1)}, ${p[1].toFixed(1)}, ${p[2].toFixed(1)} m`);
  const v = ent.vel;
  const speed = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2).toFixed(2);
  html += row("Speed",   `${speed} m/s`);
  html += row("Heading", `${ent.heading.toFixed(1)}°`);

  if (ent.type === "UAV") {
    html += `<div class="section-hdr">Flight</div>`;
    html += row("Mode",       ent.autopilot_mode || "—");
    html += row("Role",       ent.deconflict_role || "—");
    html += row("Endurance",  `${ent.endurance_s}s`,  ent.low_fuel ? "lf" : "");
    html += row("Low Fuel",   ent.low_fuel  ? "YES" : "No",  ent.low_fuel  ? "lf" : "");
    html += row("NFZ Viol.",  ent.nfz_violated ? "YES" : "No", ent.nfz_violated ? "danger" : "");

    if (ent.fov) {
      html += `<div class="section-hdr">Payload</div>`;
      html += row("Gimbal Mode", ent.fov.mode || "—");
      html += row("Gimbal Az",   `${ent.fov.gimbal_az}°`);
      html += row("Gimbal El",   `${ent.fov.gimbal_el}°`);
      html += row("FOV",         `${(ent.fov.half_angle * 2).toFixed(1)}°`);
    }
  }

  body.innerHTML = html;
  document.getElementById("panel").classList.add("visible");
}

function _hidePanel() {
  selectedId = null;
  document.getElementById("panel").classList.remove("visible");
}

// ── Filter checkboxes ─────────────────────────────────────────────────────
const _filterMap = {
  "f-uav":     "uav",
  "f-wheeled": "wheeled",
  "f-tracked": "tracked",
  "f-ped":     "ped",
  "f-tracks":  "tracks",
  "f-fov":     "fov",
  "f-roads":   "roads",
};
for (const [id, key] of Object.entries(_filterMap)) {
  document.getElementById(id).addEventListener("change", e => {
    opts[key] = e.target.checked;
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  });
}

// ── Button handlers ───────────────────────────────────────────────────────

document.getElementById("btn-new").addEventListener("click", async () => {
  const res  = await fetch("/scenarios", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: "{}" });
  const data = await res.json();
  resetView();
  connect(data.scenario_id);
});

document.getElementById("btn-pause").addEventListener("click", () => {
  paused = !paused;
  sendCommand(paused ? "pause" : "resume");
  document.getElementById("btn-pause").textContent = paused ? "▶ Resume" : "⏸ Pause";
  document.getElementById("paused-banner").classList.toggle("visible", paused);
});

document.getElementById("btn-view").addEventListener("click", () => {
  c4iActive = !c4iActive;
  toggleC4IView();
  document.getElementById("btn-view").textContent = c4iActive ? "🌍 World View" : "📡 C4I View";
  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
});

document.getElementById("btn-reset").addEventListener("click", () => {
  resetView();
  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
});

document.getElementById("panel-close").addEventListener("click", _hidePanel);

// ── Keyboard ──────────────────────────────────────────────────────────────
window.addEventListener("keydown", e => {
  if (e.key === " ") {
    e.preventDefault();
    document.getElementById("btn-pause").click();
  } else if (e.key === "r" || e.key === "R") {
    resetView();
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  } else if (e.key === "v" || e.key === "V") {
    document.getElementById("btn-view").click();
  } else if (e.key === "Escape") {
    _hidePanel();
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  } else if (e.key === "ArrowLeft") {
    vs.offsetX -= vs.worldX * 0.10 / vs.zoom;
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  } else if (e.key === "ArrowRight") {
    vs.offsetX += vs.worldX * 0.10 / vs.zoom;
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  } else if (e.key === "ArrowUp") {
    vs.offsetY += vs.worldY * 0.10 / vs.zoom;
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  } else if (e.key === "ArrowDown") {
    vs.offsetY -= vs.worldY * 0.10 / vs.zoom;
    if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
  }
});

// ── Scroll to zoom (anchored to cursor) ──────────────────────────────────
canvas.addEventListener("wheel", e => {
  e.preventDefault();

  // World point under cursor BEFORE zoom change
  const [wx, wy] = canvasToWorld(e.offsetX, e.offsetY, vs, canvas);

  const factor   = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  vs.zoom        = Math.min(20, Math.max(0.08, vs.zoom * factor));

  // Recompute offset so that (wx, wy) stays under the cursor
  const ppm      = Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
  vs.offsetX     = wx - e.offsetX / ppm;
  vs.offsetY     = wy - (canvas.height - e.offsetY) / ppm;

  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
}, { passive: false });

// ── Right-drag to pan ────────────────────────────────────────────────────
canvas.addEventListener("mousedown", e => {
  if (e.button === 2) {
    _dragging   = true;
    _dragStartX = e.clientX;
    _dragStartY = e.clientY;
    _dragOffX   = vs.offsetX;
    _dragOffY   = vs.offsetY;
  }
});

canvas.addEventListener("mousemove", e => {
  if (!_dragging) return;
  const ppm    = _pixelsPerMetre();
  const dx     = (e.clientX - _dragStartX) / ppm;
  const dy     = (e.clientY - _dragStartY) / ppm;
  vs.offsetX   = _dragOffX - dx;
  vs.offsetY   = _dragOffY + dy;
  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
});

window.addEventListener("mouseup", e => {
  if (e.button === 2) _dragging = false;
});
canvas.addEventListener("contextmenu", e => e.preventDefault());

// ── Click to select entity ────────────────────────────────────────────────
canvas.addEventListener("click", e => {
  if (!lastFrame) return;

  const HIT_PX  = 20;   // click tolerance in canvas pixels
  let   bestEnt = null;
  let   bestD2  = HIT_PX * HIT_PX;

  for (const ent of lastFrame.entities) {
    const [ex, ey] = worldToCanvas(ent.pos[0], ent.pos[1], vs, canvas);
    const d2 = (e.offsetX - ex) ** 2 + (e.offsetY - ey) ** 2;
    if (d2 < bestD2) { bestD2 = d2; bestEnt = ent; }
  }

  if (bestEnt) {
    _showPanel(bestEnt);
  } else {
    _hidePanel();
  }

  if (lastFrame) drawFrame(canvas, lastFrame, vs, opts, selectedId);
});

// ── Initial draw (empty frame) ────────────────────────────────────────────
drawFrame(canvas, { t: 0, step: 0, entities: [], tracks: [] }, vs, opts, null);
document.getElementById("hud-status").textContent = "Click 'New Scenario'";
