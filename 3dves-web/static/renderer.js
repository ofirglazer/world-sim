/**
 * renderer.js
 * ===========
 * Canvas 2D renderer for the 3DVES simulation.
 *
 * Exports one function: drawFrame(canvas, frame, viewState, opts, selectedId).
 * Everything else is private to this module.
 *
 * Draw order (back to front):
 *   1.  Background fill
 *   2.  Road network  (both views; drawn directly every frame — no cache)
 *   3.  NFZ cylinders
 *   4.  Track covariance ellipses
 *   5.  FOV cones      (all views)
 *   6.  Waypoint path  (world view only; selected entity only)
 *   7.  Entities       (all views; C4I = UAVs + detected only)
 *   8.  Entity labels  (world view only; hidden when zoomed out)
 *   9.  Selected entity highlight
 *  10.  HUD DOM update
 *
 * Road network — no offscreen cache
 * ----------------------------------
 * The road network is drawn directly onto the main canvas each frame
 * using worldToCanvas(), which incorporates the current offsetX/Y and
 * zoom.  A previous implementation cached the network in an OffscreenCanvas
 * keyed only on zoom and view mode, causing roads to stay fixed while the
 * rest of the scene panned.  At 180 edges the direct draw costs < 0.5 ms
 * per frame so caching provides no meaningful benefit and is removed.
 *
 * Coordinate system
 * -----------------
 * World: origin bottom-left, Y increases upward.
 * Canvas: origin top-left, Y increases downward.
 * worldToCanvas() applies the flip and the zoom/pan transform.
 *
 * View modes
 * ----------
 * World view  : full scene — all entity types, road network (amber),
 *               labels, waypoints for selected entity.
 * C4I view    : tactical overlay — UAVs and detected entities only,
 *               near-black background, road network (dim green),
 *               no labels, no waypoints.
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
  WHEELED_VEHICLE:  6,
  TRACKED_VEHICLE:  7,
  PEDESTRIAN:       4,
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

// Waypoint colours — high contrast against dark world background (#2e3440)
const WP_LINE_WORLD = "rgba(180,220,80,0.90)";   // yellow-green dashed line
const WP_DOT_WORLD  = "rgba(180,220,80,1.00)";   // fully opaque intermediate dots
const WP_DEST_WORLD = "rgba(0,229,255,1.00)";    // cyan — current destination

// Road colours per view mode
const ROAD_EDGE_WORLD = "rgba(200,170,80,0.80)";  // warm amber on dark slate
const ROAD_NODE_WORLD = "rgba(200,170,80,1.00)";
const ROAD_EDGE_C4I   = "rgba(60,110,60,0.70)";   // dim military green on near-black
const ROAD_NODE_C4I   = "rgba(60,110,60,0.90)";

// ── Module state ──────────────────────────────────────────────────────────
let _roadData = null;    // {nodes: {id:[x,y]}, edges:[[a,b],...]} from hello frame
let _nfzData  = [];      // [{cx,cy,radius_m,alt_max_m}, ...] from hello frame
let _c4iView  = true;    // start in tactical C4I mode

// Detection flash: entity_id → timestamp (ms) of last detected frame
const _detectionTimes = new Map();
const FLASH_DURATION_MS = 600;

// ── Coordinate helpers ────────────────────────────────────────────────────

/**
 * World metres → canvas pixels.
 * Incorporates offsetX/Y (pan) and zoom so roads, entities, and
 * waypoints all share the same transform.
 *
 * @param {number} wx  World X (m, origin left).
 * @param {number} wy  World Y (m, origin bottom — Y is flipped for canvas).
 * @param {object} vs  {offsetX, offsetY, zoom, worldX, worldY}.
 * @param {HTMLCanvasElement} canvas
 * @returns {[number, number]} [canvasX, canvasY]
 */
export function worldToCanvas(wx, wy, vs, canvas) {
  const ppm = _ppm(vs, canvas);
  return [
    (wx - vs.offsetX) * ppm,
    canvas.height - (wy - vs.offsetY) * ppm,
  ];
}

/** Canvas pixels → world metres (inverse of worldToCanvas). */
export function canvasToWorld(cx, cy, vs, canvas) {
  const ppm = _ppm(vs, canvas);
  return [
    cx / ppm + vs.offsetX,
    (canvas.height - cy) / ppm + vs.offsetY,
  ];
}

/** Pixels per metre for the current view state. */
function _ppm(vs, canvas) {
  return Math.min(canvas.width, canvas.height) /
         Math.max(vs.worldX, vs.worldY) * vs.zoom;
}

/** World distance (m) → canvas pixels (scalar). */
function worldToCanvasDist(dm, vs, canvas) {
  return dm * _ppm(vs, canvas);
}

// ── Covariance eigen-decomposition for track ellipses ────────────────────

function _eigenEllipse(cov) {
  const a  = cov[0][0], b = cov[0][1], d = cov[1][1];
  const tr   = a + d;
  const det  = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1   = tr / 2 + disc;
  const l2   = tr / 2 - disc;
  const rx   = 2 * Math.sqrt(Math.max(0, l1));   // 2-sigma semi-axis
  const ry   = 2 * Math.sqrt(Math.max(0, l2));
  const angle = (Math.abs(b) < 1e-9 && a >= d) ? 0
              : (Math.abs(b) < 1e-9 && a <  d) ? Math.PI / 2
              : Math.atan2(l1 - a, b);
  return { rx, ry, angle };
}

// ── Utilities ─────────────────────────────────────────────────────────────

function _fmtTime(s) {
  const m  = Math.floor(s / 60).toString().padStart(2, "0");
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
}

// ── Public helpers called by client.js ────────────────────────────────────

export function toggleC4IView() {
  _c4iView = !_c4iView;
}

export function setRoadData(data) {
  _roadData = data;
}

export function setNfzData(data) {
  _nfzData = data || [];
}

// ── Road network drawing ──────────────────────────────────────────────────

/**
 * Draw the road network directly onto *ctx* using the current view state.
 *
 * Called every frame inside drawFrame() so pan and zoom are always
 * applied correctly — there is no stale offscreen canvas to worry about.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} vs    Current view state.
 * @param {HTMLCanvasElement} canvas
 */
function _drawRoads(ctx, vs, canvas) {
  if (!_roadData) return;

  const edgeCol = _c4iView ? ROAD_EDGE_C4I : ROAD_EDGE_WORLD;
  const nodeCol = _c4iView ? ROAD_NODE_C4I : ROAD_NODE_WORLD;
  const lw      = Math.max(1, worldToCanvasDist(1.5, vs, canvas));
  const nr      = Math.max(2, worldToCanvasDist(2.5, vs, canvas));

  // ── Edges ────────────────────────────────────────────────────────────
  ctx.strokeStyle = edgeCol;
  ctx.lineWidth   = lw;
  for (const [aId, bId] of (_roadData.edges || [])) {
    const a = _roadData.nodes[aId];
    const b = _roadData.nodes[bId];
    if (!a || !b) continue;
    const [ax, ay] = worldToCanvas(a[0], a[1], vs, canvas);
    const [bx, by] = worldToCanvas(b[0], b[1], vs, canvas);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // ── Nodes ────────────────────────────────────────────────────────────
  ctx.fillStyle = nodeCol;
  for (const pos of Object.values(_roadData.nodes || {})) {
    const [cx2, cy2] = worldToCanvas(pos[0], pos[1], vs, canvas);
    ctx.beginPath();
    ctx.arc(cx2, cy2, nr, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── Waypoint path drawing ─────────────────────────────────────────────────

/**
 * Draw the planned path for a selected entity (world view only).
 *
 * Renders a dashed polyline through remaining waypoints, a small dot
 * at each intermediate point, and a contrasting ring+dot at
 * current_destination.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} ent    Entity record from the frame.
 * @param {object} vs     View state.
 * @param {HTMLCanvasElement} canvas
 */
function _drawWaypoints(ctx, ent, vs, canvas) {
  const wps  = ent.waypoints || [];
  const dest = ent.current_destination;
  if (wps.length === 0 && !dest) return;

  const [ex, ey] = worldToCanvas(ent.pos[0], ent.pos[1], vs, canvas);
  const points   = wps.map(wp => worldToCanvas(wp[0], wp[1], vs, canvas));

  // Dashed polyline from entity position through all remaining waypoints
  if (points.length > 0) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = WP_LINE_WORLD;
    ctx.lineWidth   = 2.5; // 1.5
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    for (const [px, py] of points) ctx.lineTo(px, py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // Intermediate waypoint dots
  const dotR = Math.max(3, worldToCanvasDist(3, vs, canvas));
  ctx.fillStyle = WP_DOT_WORLD;
  for (const [px, py] of points) {
    ctx.beginPath();
    ctx.arc(px, py, dotR, 0, Math.PI * 2);
    ctx.fill();
  }

  // Current destination — larger cyan marker
  if (dest) {
    const [dx, dy] = worldToCanvas(dest[0], dest[1], vs, canvas);
    const destR    = Math.max(5, worldToCanvasDist(5, vs, canvas));
    ctx.beginPath();
    ctx.arc(dx, dy, destR + 3, 0, Math.PI * 2);
    ctx.strokeStyle = WP_DEST_WORLD;
    ctx.lineWidth   = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(dx, dy, destR, 0, Math.PI * 2);
    ctx.fillStyle = WP_DEST_WORLD;
    ctx.fill();
  }
}

// ── Main draw function ────────────────────────────────────────────────────

/**
 * Draw one simulation frame onto *canvas*.
 *
 * @param {HTMLCanvasElement}  canvas
 * @param {object}             frame       Deserialised frame from server.
 * @param {object}             vs          View state {offsetX,offsetY,zoom,...}.
 * @param {object}             opts        Filter flags from checkboxes.
 * @param {string|null}        selectedId  Currently selected entity ID.
 */
export function drawFrame(canvas, frame, vs, opts, selectedId) {
  const ctx = canvas.getContext("2d");
  const now = performance.now();

  // Resize canvas to CSS dimensions (handles window resize)
  if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
    canvas.width  = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }

  // ── 1. Background ─────────────────────────────────────────────────
  ctx.fillStyle = _c4iView ? "#0a0a1a" : "#2e3440";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // ── 2. Road network (both views, direct draw — pan/zoom correct) ──
  if (opts.roads) {
    _drawRoads(ctx, vs, canvas);
  }

  // ── 3. NFZ cylinders (from hello frame module state) ─────────────────
  if (_nfzData.length > 0) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(255,23,68,0.70)";
    ctx.lineWidth   = 1.5;
    for (const nfz of _nfzData) {
      const [cx2, cy2] = worldToCanvas(nfz.cx, nfz.cy, vs, canvas);
      const r = worldToCanvasDist(nfz.radius_m, vs, canvas);
      ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ── 4. Track covariance ellipses ──────────────────────────────────
  if (opts.tracks && frame.tracks) {
    for (const trk of frame.tracks) {
      if (_c4iView && !trk.is_eoi) continue;

      const col = TRACK_COLOUR[trk.quality] || "#888";
      const { rx, ry, angle } = _eigenEllipse(trk.cov);
      const [tx, ty] = worldToCanvas(trk.pos_xy[0], trk.pos_xy[1], vs, canvas);
      const rxPx = Math.min(80, worldToCanvasDist(rx, vs, canvas));
      const ryPx = Math.min(80, worldToCanvasDist(ry, vs, canvas));

      ctx.save();
      ctx.translate(tx, ty);
      ctx.rotate(-angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(2, rxPx), Math.max(2, ryPx), 0, 0, Math.PI * 2);
      ctx.fillStyle   = col + "40";
      ctx.strokeStyle = col + "cc";
      ctx.lineWidth   = 1;
      ctx.fill(); ctx.stroke();
      ctx.restore();

      ctx.beginPath();
      ctx.arc(tx, ty, 3, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    }
  }

  // ── 5. FOV cones (all views) ───────────────────────────────────────
  if (opts.fov) {
    for (const ent of frame.entities) {
      if (ent.type !== "UAV" || !ent.fov || !ent.fov.tip || !ent.fov.axis) continue;
      const mode    = ent.fov.mode || "SCAN";
      const halfAng = ent.fov.half_angle * Math.PI / 180;
      const [tx2, ty2] = worldToCanvas(ent.fov.tip[0], ent.fov.tip[1], vs, canvas);
      const axisAng = Math.atan2(-ent.fov.axis[1], ent.fov.axis[0]);
      const alt     = ent.pos[2];
      const el_rad  = ent.fov.gimbal_el * Math.PI / 180;
      const gndDist = Math.abs(alt / Math.tan(el_rad + 1e-6));
      const coneLen = Math.min(300, worldToCanvasDist(gndDist, vs, canvas));

      ctx.save();
      ctx.translate(tx2, ty2);
      ctx.rotate(axisAng);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(coneLen * Math.cos(-halfAng), coneLen * Math.sin(-halfAng));
      ctx.lineTo(coneLen * Math.cos( halfAng), coneLen * Math.sin( halfAng));
      ctx.closePath();
      ctx.fillStyle   = FOV_COLOUR[mode]  || "rgba(0,229,255,0.15)";
      ctx.strokeStyle = FOV_STROKE[mode]  || "rgba(0,229,255,0.55)";
      ctx.lineWidth   = 1;
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }
  }

  // ── 6. Waypoint path (world view only, selected entity only) ──────
  if (!_c4iView && selectedId) {
    const selEnt = frame.entities.find(e => e.id === selectedId);
    if (selEnt) _drawWaypoints(ctx, selEnt, vs, canvas);
  }

  // ── 7. Entities ───────────────────────────────────────────────────
  const ppm       = _ppm(vs, canvas);
  const showLabel = !_c4iView && vs.zoom > 0.4;

  for (const ent of frame.entities) {
    // C4I: show only UAVs and entities with an active detection flash
    if (_c4iView) {
      if (ent.detected) _detectionTimes.set(ent.id, now);
      const flashAge = now - (_detectionTimes.get(ent.id) || 0);
      if (ent.type !== "UAV" && flashAge >= FLASH_DURATION_MS) continue;
    }

    // World view: honour type-filter checkboxes
    if (!_c4iView) {
      if (ent.type === "UAV"             && !opts.uav)     continue;
      if (ent.type === "WHEELED_VEHICLE" && !opts.wheeled) continue;
      if (ent.type === "TRACKED_VEHICLE" && !opts.tracked) continue;
      if (ent.type === "PEDESTRIAN"      && !opts.ped)     continue;
    }

    const [ex, ey] = worldToCanvas(ent.pos[0], ent.pos[1], vs, canvas);
    const rWorld   = TYPE_RADIUS_WORLD[ent.type] || 5;
    const r        = Math.max(3, rWorld * ppm);
    const baseCol  = TYPE_COLOUR[ent.type] || "#aaa";

    // Detection flash ring (yellow, fading over FLASH_DURATION_MS)
    if (ent.detected) _detectionTimes.set(ent.id, now);
    const flashAge2 = now - (_detectionTimes.get(ent.id) || 0);
    if (flashAge2 < FLASH_DURATION_MS) {
      const alpha  = 1 - flashAge2 / FLASH_DURATION_MS;
      ctx.beginPath(); ctx.arc(ex, ey, r + 6 * alpha, 0, Math.PI * 2);
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

    // Heading tick
    if (!_c4iView || ent.type === "UAV") {
      const hdRad = ent.heading * Math.PI / 180;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + Math.cos(hdRad) * (r + 5),
                 ey - Math.sin(hdRad) * (r + 5));
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
    }

    // ── 8. Label (world view only) ────────────────────────────────
    if (showLabel) {
      const shortId = ent.id.length > 8 ? ent.id.slice(0, 8) : ent.id;
      ctx.font      = `${Math.max(9, 10 * vs.zoom)}px monospace`;
      ctx.fillStyle = "rgba(200,200,200,0.75)";
      ctx.fillText(shortId, ex + r + 2, ey - 2);
    }

    // ── 9. Selected entity highlight ──────────────────────────────
    if (ent.id === selectedId) {
      ctx.beginPath(); ctx.arc(ex, ey, r + 7, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 2.5;
      ctx.stroke();
    }
  }

  // ── 10. HUD DOM update ────────────────────────────────────────────
  document.getElementById("hud-time").textContent   = _fmtTime(frame.t);
  document.getElementById("hud-step").textContent   = frame.step;
  document.getElementById("hud-alive").textContent  = frame.entities.length;
  document.getElementById("hud-tracks").textContent = frame.tracks ? frame.tracks.length : 0;
}
