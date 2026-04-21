/**
 * renderer.js
 * ===========
 * Canvas 2D renderer for the 3DVES simulation.
 *
 * Exports one function: drawFrame(canvas, frame, viewState, opts).
 * Everything else is private to this module.
 *
 * Draw order (back to front):
 *   1. Background fill
 *   2. Road network  (cached offscreen canvas, redrawn on zoom change)
 *   3. NFZ cylinders
 *   4. Track covariance ellipses
 *   5. FOV cones
 *   6. Entities
 *   7. Entity labels  (hidden when zoomed out)
 *   8. Selected entity highlight
 *
 * Coordinate system
 * -----------------
 * World: origin bottom-left, Y increases upward.
 * Canvas: origin top-left, Y increases downward.
 * worldToCanvas() applies the flip and the zoom/pan transform.
 *
 * No external dependencies — plain Canvas 2D API only.
 */

// ── Entity colours ────────────────────────────────────────────────────────
const TYPE_COLOUR = {
  UAV:             "#b39ddb",
  WHEELED_VEHICLE: "#66bb6a",
  TRACKED_VEHICLE: "#ffa726",
  PEDESTRIAN:      "#42a5f5",
};

const TYPE_RADIUS_WORLD = {   // entity dot radius in world metres
  UAV:             12,
  WHEELED_VEHICLE: 6,
  TRACKED_VEHICLE: 7,
  PEDESTRIAN:      4,
};

// Track quality colours (match TRK_VIZ_* in defaults.py)
const TRACK_COLOUR = {
  HIGH:   "#00E676",
  MEDIUM: "#FFEA00",
  LOW:    "#FF6D00",
};

// FOV cone colours by gimbal mode
const FOV_COLOUR = {
  SCAN:  "rgba(0,229,255,0.15)",
  STARE: "rgba(255,234,0,0.18)",
  CUED:  "rgba(255,109,0,0.20)",
};
const FOV_STROKE = {
  SCAN:  "rgba(0,229,255,0.55)",
  STARE: "rgba(255,234,0,0.65)",
  CUED:  "rgba(255,109,0,0.70)",
};

// ── Road network cache ────────────────────────────────────────────────────
let _roadCache      = null;   // OffscreenCanvas
let _roadCacheZoom  = null;   // zoom level when cache was built
let _roadData       = null;   // {nodes, edges} from the hello frame or POST

// ── Detection flash state ─────────────────────────────────────────────────
// Maps entity_id → timestamp (ms) of last detection
const _detectionTimes = new Map();
const FLASH_DURATION_MS = 600;

// ── C4I / World view toggle ───────────────────────────────────────────────
let _c4iView = true;   // start in tactical C4I mode

// ── Coordinate helpers ────────────────────────────────────────────────────

/**
 * Convert world metres → canvas pixels.
 * @param {number} wx  World X (metres, origin left).
 * @param {number} wy  World Y (metres, origin bottom).
 * @param {object} vs  viewState {offsetX, offsetY, zoom, worldX, worldY}.
 * @param {HTMLCanvasElement} canvas
 * @returns {[number, number]} [cx, cy] canvas pixel coordinates.
 */
export function worldToCanvas(wx, wy, vs, canvas) {
  const ppm = Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
  const cx  = (wx - vs.offsetX) * ppm;
  const cy  = canvas.height - (wy - vs.offsetY) * ppm;
  return [cx, cy];
}

/**
 * Convert canvas pixels → world metres (inverse of worldToCanvas).
 */
export function canvasToWorld(cx, cy, vs, canvas) {
  const ppm = Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
  const wx  = cx / ppm + vs.offsetX;
  const wy  = (canvas.height - cy) / ppm + vs.offsetY;
  return [wx, wy];
}

/** World metres → canvas pixels (scalar, for radius/size conversions). */
function worldToCanvasDist(dm, vs, canvas) {
  const ppm = Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
  return dm * ppm;
}

// ── Offscreen road cache ──────────────────────────────────────────────────

function _buildRoadCache(vs, canvas) {
  const oc  = new OffscreenCanvas(canvas.width, canvas.height);
  const ctx = oc.getContext("2d");

  if (!_roadData) { _roadCache = oc; _roadCacheZoom = vs.zoom; return; }

  const edgeCol  = _c4iView ? "rgba(80,120,80,0.55)" : "rgba(160,140,80,0.70)";
  const nodeCol  = _c4iView ? "rgba(80,120,80,0.75)" : "rgba(160,140,80,0.90)";

  ctx.strokeStyle = edgeCol;
  ctx.lineWidth   = Math.max(1, worldToCanvasDist(1.5, vs, canvas));

  for (const [aId, bId] of (_roadData.edges || [])) {
    const a = _roadData.nodes[aId];
    const b = _roadData.nodes[bId];
    if (!a || !b) continue;
    const [ax, ay] = worldToCanvas(a[0], a[1], vs, canvas);
    const [bx, by] = worldToCanvas(b[0], b[1], vs, canvas);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }

  ctx.fillStyle = nodeCol;
  const nr = Math.max(2, worldToCanvasDist(2.5, vs, canvas));
  for (const [, pos] of Object.entries(_roadData.nodes || {})) {
    const [cx2, cy2] = worldToCanvas(pos[0], pos[1], vs, canvas);
    ctx.beginPath(); ctx.arc(cx2, cy2, nr, 0, Math.PI * 2); ctx.fill();
  }

  _roadCache     = oc;
  _roadCacheZoom = vs.zoom;
}

// ── 2×2 eigen-decomposition for track ellipses ───────────────────────────

function _eigenEllipse(cov) {
  // cov = [[a,b],[b,d]]
  const a  = cov[0][0], b = cov[0][1], d = cov[1][1];
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  // Semi-axes: 2-sigma
  const rx = 2 * Math.sqrt(Math.max(0, l1));
  const ry = 2 * Math.sqrt(Math.max(0, l2));
  // Rotation angle of the major axis
  const angle = (Math.abs(b) < 1e-9 && a >= d) ? 0
              : (Math.abs(b) < 1e-9 && a <  d) ? Math.PI / 2
              : Math.atan2(l1 - a, b);
  return { rx, ry, angle };
}

// ── Utility ───────────────────────────────────────────────────────────────

function _fmtTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ── Toggle helpers (called from client.js) ────────────────────────────────

export function toggleC4IView() {
  _c4iView = !_c4iView;
  _roadCache = null;   // invalidate road cache — colours change
}

export function setRoadData(data) {
  _roadData  = data;
  _roadCache = null;
}

// ── Main draw function ────────────────────────────────────────────────────

/**
 * Draw one simulation frame onto *canvas*.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} frame          Deserialised frame from server.
 * @param {object} vs             View state {offsetX, offsetY, zoom, worldX, worldY}.
 * @param {object} opts           Display options from filter checkboxes.
 * @param {string|null} selectedId  Currently selected entity ID (or null).
 */
export function drawFrame(canvas, frame, vs, opts, selectedId) {
  const ctx  = canvas.getContext("2d");
  const now  = performance.now();

  // Resize canvas to match CSS size (handles window resize)
  if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
    canvas.width  = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    _roadCache    = null;   // invalidate on resize
  }

  // ── 1. Background ──────────────────────────────────────────────────────
  ctx.fillStyle = _c4iView ? "#0a0a1a" : "#eae8dc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // ── 2. Road network ────────────────────────────────────────────────────
  if (opts.roads) {
    if (!_roadCache || _roadCacheZoom !== vs.zoom) {
      _buildRoadCache(vs, canvas);
    }
    ctx.drawImage(_roadCache, 0, 0);
  }

  // ── 3. NFZ cylinders ──────────────────────────────────────────────────
  // NFZ data comes from the server hello frame if present
  if (frame.nfzs) {
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(255,23,68,0.70)";
    ctx.lineWidth   = 1.5;
    for (const nfz of frame.nfzs) {
      const [cx2, cy2] = worldToCanvas(nfz.cx, nfz.cy, vs, canvas);
      const r = worldToCanvasDist(nfz.radius_m, vs, canvas);
      ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // ── 4. Track covariance ellipses ──────────────────────────────────────
  if (opts.tracks && frame.tracks) {
    for (const trk of frame.tracks) {
      const col = TRACK_COLOUR[trk.quality] || "#888";
      const { rx, ry, angle } = _eigenEllipse(trk.cov);
      const [tx, ty] = worldToCanvas(trk.pos_xy[0], trk.pos_xy[1], vs, canvas);
      const rxPx = Math.min(80, worldToCanvasDist(rx, vs, canvas));
      const ryPx = Math.min(80, worldToCanvasDist(ry, vs, canvas));

      ctx.save();
      ctx.translate(tx, ty);
      ctx.rotate(-angle);   // canvas Y is flipped, negate rotation
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(2, rxPx), Math.max(2, ryPx), 0, 0, Math.PI * 2);
      ctx.fillStyle   = col + "40";   // 25% alpha fill
      ctx.strokeStyle = col + "cc";
      ctx.lineWidth   = 1;
      ctx.fill(); ctx.stroke();
      ctx.restore();

      // Centroid dot
      ctx.beginPath();
      ctx.arc(tx, ty, 3, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    }
  }

  // ── 5. FOV cones ──────────────────────────────────────────────────────
  if (opts.fov) {
    for (const ent of frame.entities) {
      if (ent.type !== "UAV" || !ent.fov || !ent.fov.tip || !ent.fov.axis) continue;

      const mode   = ent.fov.mode || "SCAN";
      const halfAng = ent.fov.half_angle * Math.PI / 180;
      const [tx2, ty2] = worldToCanvas(ent.fov.tip[0], ent.fov.tip[1], vs, canvas);

      // Aim direction in canvas space (Y flipped)
      const axisAng = Math.atan2(-ent.fov.axis[1], ent.fov.axis[0]);

      // Cone length: project altitude through elevation
      const alt      = ent.pos[2];
      const el_rad   = ent.fov.gimbal_el * Math.PI / 180;
      const gndDist  = Math.abs(alt / Math.tan(el_rad + 1e-6));
      const coneLenPx = Math.min(300, worldToCanvasDist(gndDist, vs, canvas));

      ctx.save();
      ctx.translate(tx2, ty2);
      ctx.rotate(axisAng);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(coneLenPx * Math.cos(-halfAng), coneLenPx * Math.sin(-halfAng));
      ctx.lineTo(coneLenPx * Math.cos( halfAng), coneLenPx * Math.sin( halfAng));
      ctx.closePath();
      ctx.fillStyle   = FOV_COLOUR[mode]  || "rgba(0,229,255,0.15)";
      ctx.strokeStyle = FOV_STROKE[mode]  || "rgba(0,229,255,0.55)";
      ctx.lineWidth   = 1;
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }
  }

  // ── 6 & 7. Entities and labels ────────────────────────────────────────
  const ppm      = Math.min(canvas.width, canvas.height) / Math.max(vs.worldX, vs.worldY) * vs.zoom;
  const showLabel = vs.zoom > 0.4;

  for (const ent of frame.entities) {
    // Filter by type visibility
    if (ent.type === "UAV"             && !opts.uav)     continue;
    if (ent.type === "WHEELED_VEHICLE" && !opts.wheeled) continue;
    if (ent.type === "TRACKED_VEHICLE" && !opts.tracked) continue;
    if (ent.type === "PEDESTRIAN"      && !opts.ped)     continue;

    const [ex, ey] = worldToCanvas(ent.pos[0], ent.pos[1], vs, canvas);
    const rWorld   = TYPE_RADIUS_WORLD[ent.type] || 5;
    const r        = Math.max(3, rWorld * ppm);
    const baseCol  = TYPE_COLOUR[ent.type] || "#aaa";

    // Detection flash (yellow ring fading over FLASH_DURATION_MS)
    if (ent.detected) _detectionTimes.set(ent.id, now);
    const lastDet  = _detectionTimes.get(ent.id) || 0;
    const flashAge = now - lastDet;
    if (flashAge < FLASH_DURATION_MS) {
      const alpha  = 1 - flashAge / FLASH_DURATION_MS;
      const flashR = r + 6 * alpha;
      ctx.beginPath(); ctx.arc(ex, ey, flashR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,234,0,${alpha.toFixed(2)})`;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    // EOI outer ring
    if (ent.is_eoi) {
      ctx.beginPath(); ctx.arc(ex, ey, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    }

    // Main dot
    ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2);
    ctx.fillStyle = baseCol;
    ctx.fill();

    // Heading tick (shows direction of travel)
    if (ent.type !== "PEDESTRIAN" || vs.zoom > 0.8) {
      const hdRad = ent.heading * Math.PI / 180;
      const hx    = ex + Math.cos(hdRad) * (r + 5);
      const hy    = ey - Math.sin(hdRad) * (r + 5);
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx, hy);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
    }

    // Label
    if (showLabel) {
      const shortId = ent.id.slice(0, 8);
      ctx.font      = `${Math.max(9, 10 * vs.zoom)}px var(--font-mono, monospace)`;
      ctx.fillStyle = _c4iView ? "rgba(200,200,200,0.75)" : "rgba(40,40,40,0.80)";
      ctx.fillText(shortId, ex + r + 2, ey - 2);
    }

    // Selected entity highlight
    if (ent.id === selectedId) {
      ctx.beginPath(); ctx.arc(ex, ey, r + 7, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 2.5;
      ctx.stroke();
    }
  }

  // ── 8. HUD update (DOM, not canvas) ───────────────────────────────────
  document.getElementById("hud-time").textContent  = _fmtTime(frame.t);
  document.getElementById("hud-step").textContent  = frame.step;
  document.getElementById("hud-alive").textContent = frame.entities.length;
  document.getElementById("hud-tracks").textContent = frame.tracks ? frame.tracks.length : 0;
}
