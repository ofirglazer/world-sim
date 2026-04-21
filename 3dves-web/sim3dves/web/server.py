"""
sim3dves.web.server
===================
FastAPI application: WebSocket broadcast server for the 3DVES simulation.

Routes
------
POST   /scenarios              Create and start a new simulation scenario.
DELETE /scenarios/{id}         Stop and destroy a scenario.
GET    /scenarios/{id}/info    Poll worker liveness and world extents.
WS     /ws/sim/{id}            Subscribe to the frame stream; send commands.
GET    /                       Serve static/ (index.html + JS).

Broadcast loop
--------------
An asyncio background task (_broadcast_loop) drains every session's
frame_queue and fans out JSON frames to all connected WebSocket clients.
It uses asyncio.get_event_loop().run_in_executor to bridge the
synchronous queue.get_nowait() call to the async event loop without
blocking it.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
# TODO The --reload flag watches for source changes and restarts automatically during development. Remove it in production.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import queue
from pathlib import Path
from typing import Set

import numpy as np
import uuid as _uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.event_bus import EventType
from sim3dves.core.world import NFZCylinder, World
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.uav import SearchPattern, UAVEntity
from sim3dves.entities.vehicle import (
    TrackedVehicleEntity,
    VehicleKinematics,
    WheeledVehicleEntity,
)
from sim3dves.maps.road_network import RoadNetwork
from sim3dves.payload.cueing_policy import HighQualityEoiCueingPolicy
from sim3dves.payload.optical_payload import OpticalPayload
from sim3dves.web.session_registry import registry

_D = SimDefaults()

# ---------------------------------------------------------------------------
# Scenario dimensions (match run_simulation.py defaults)
# ---------------------------------------------------------------------------
_WORLD_X:       float = 1500.0
_WORLD_Y:       float = 1500.0
_GRID_ROWS:     int   = 10
_GRID_COLS:     int   = 10

@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    """
    FastAPI lifespan handler (replaces deprecated @app.on_event).

    Starts the frame-broadcast background task when the server starts
    and ensures it is cancelled cleanly on shutdown.
    """
    task = asyncio.create_task(_broadcast_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="3DVES Web Service", version="M5", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Fan-out registry of WebSocket connections for one scenario session.

    Design Pattern: Observer -- clients subscribe to the frame stream;
    the broadcaster drains the worker queue and fans out to all sockets.
    """

    def __init__(self) -> None:
        self._sockets: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a WebSocket connection."""
        await ws.accept()
        self._sockets.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection from the registry."""
        self._sockets.discard(ws)

    async def broadcast(self, text: str) -> None:
        """
        Fan out *text* to all connected clients.

        Removes any client whose connection has dropped silently
        (no WebSocketDisconnect raised for network-level drops).
        """
        dead: Set[WebSocket] = set()
        for ws in self._sockets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        self._sockets -= dead

    @property
    def count(self) -> int:
        """Number of currently connected clients."""
        return len(self._sockets)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_default_engine(seed: int = _D.SIM_SEED) -> SimulationEngine:
    """
    Build and return a fully populated SimulationEngine.

    Mirrors run_simulation.py::main() exactly so the web service runs
    the identical M5 scenario.  The engine is returned without starting
    its step loop -- the SimulationWorker owns that.

    Parameters
    ----------
    seed : int
        RNG seed for deterministic entity spawning (SIM-003).

    Returns
    -------
    SimulationEngine
        Configured, populated, ready for worker.start().
    """
    rng = np.random.default_rng(seed)

    # Road network and world
    origin = np.array(_D.GRID_ORIGIN, dtype=float)
    road_network = RoadNetwork.build_grid(
        _GRID_ROWS, _GRID_COLS, _D.GRID_SPACING_M, origin,
        speed_limit_mps=_D.ROAD_SPEED_LIMIT_MPS,
    )
    world = World(
        extent=np.array([_WORLD_X, _WORLD_Y]),
        road_network=road_network,
        nfz_cylinders=[],          # NFZs disabled for default scenario
        alt_floor_m=_D.UAV_ALT_FLOOR_M,
        alt_ceil_m=_D.UAV_ALT_CEIL_M,
    )

    config = SimulationConfig(
        log_file=Path(f"sim_logs/sim_{seed}.jsonl"),
        seed=seed,
    )
    sim = SimulationEngine(config, world)

    # Wheeled vehicles
    node_ids = road_network.node_ids()
    kin_w = VehicleKinematics()
    for i in range(_D.NUM_WHEELED):
        nid = node_ids[int(rng.integers(0, len(node_ids)))]
        xy  = road_network.node_position(nid)
        pos = np.array([xy[0], xy[1], 0.0])
        sim.add_entity(WheeledVehicleEntity(
            entity_id    = str(_uuid.uuid4()),
            position     = pos,
            heading      = float(rng.uniform(0.0, 360.0)),
            road_network = road_network,
            kinematics   = kin_w,
            is_eoi       = (i % 5 == 0),
            rng          = np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        ))

    # Tracked vehicles
    for i in range(_D.NUM_TRACKED):
        xy  = rng.random(2) * world.extent
        pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
        sim.add_entity(TrackedVehicleEntity(
            entity_id    = str(_uuid.uuid4()),
            position     = pos,
            heading      = float(rng.uniform(0.0, 360.0)),
            world_extent = world.extent,
            is_eoi       = (i % 5 == 1),
            rng          = np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        ))

    # Pedestrians
    for i in range(_D.NUM_PEDESTRIANS):
        xy  = rng.random(2) * world.extent
        pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
        sim.add_entity(PedestrianEntity(
            entity_id = str(_uuid.uuid4()),
            position  = pos,
            velocity  = rng.standard_normal(3),
            is_eoi    = (i % 10 == 0),
        ))

    # UAVs with payloads
    search_patterns = [
        SearchPattern.LAWNMOWER,
        SearchPattern.EXPANDING_SPIRAL,
        SearchPattern.RANDOM_WALK,
        SearchPattern.LAWNMOWER,
    ]
    uav_entities: list[UAVEntity] = []
    for i in range(_D.NUM_UAVS):
        pos = np.array([
            float(rng.uniform(0.0, world.extent[0])),
            float(rng.uniform(0.0, world.extent[1])),
            _D.UAV_CRUISE_ALT_M,
        ])
        uav = UAVEntity(
            entity_id      = f"uav-{i:02d}",
            position       = pos,
            heading        = float(rng.uniform(0.0, 360.0)),
            launch_position= pos.copy(),
            search_pattern = search_patterns[i % len(search_patterns)],
            cruise_altitude_m = _D.UAV_CRUISE_ALT_M + i * 10.0,
            endurance_s    = _D.UAV_ENDURANCE_S,
            world_extent   = world.extent,
            alt_floor_m    = _D.UAV_ALT_FLOOR_M,
            alt_ceil_m     = _D.UAV_ALT_CEIL_M,
            nfz_cylinders  = [],
            is_eoi         = False,
            rng            = np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        )
        uav.payload = OpticalPayload(
            owner_id = uav.entity_id,
            rng      = np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        )
        sim.add_entity(uav)
        uav_entities.append(uav)

    # Cueing policy (BUG-011 fix: subscribe to TRACK_QUALITY_HIGH)
    cueing_policy = HighQualityEoiCueingPolicy(
        uav_entities  = uav_entities,
        track_manager = sim.track_manager,
    )
    sim.event_bus.subscribe(
        EventType.TRACK_QUALITY_HIGH, cueing_policy.on_track_acquired
    )

    return sim


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/scenarios")
async def create_scenario() -> dict:
    """
    Create and start a new simulation scenario.

    Returns
    -------
    dict
        scenario_id : str   UUID used for WebSocket and info endpoints.
        world_x     : float World X extent (m) for browser view init.
        world_y     : float World Y extent (m).
    """
    # Ensure log directory exists
    Path("sim_logs").mkdir(exist_ok=True)

    engine = build_default_engine()
    cm     = ConnectionManager()
    sid    = registry.create(engine, cm, world_x=_WORLD_X, world_y=_WORLD_Y)

    return {"scenario_id": sid, "world_x": _WORLD_X, "world_y": _WORLD_Y}


@app.delete("/scenarios/{scenario_id}", status_code=204)
async def delete_scenario(scenario_id: str) -> None:
    """Stop and destroy a running scenario."""
    if not registry.destroy(scenario_id):
        raise HTTPException(status_code=404, detail="Scenario not found")


@app.get("/scenarios/{scenario_id}/info")
async def scenario_info(scenario_id: str) -> dict:
    """Return liveness and world dimensions for *scenario_id*."""
    session = registry.get(scenario_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "scenario_id": scenario_id,
        "alive":       session.worker.is_alive,
        "paused":      session.worker.is_paused,
        "world_x":     session.world_x,
        "world_y":     session.world_y,
        "clients":     session.connection_manager.count,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/sim/{scenario_id}")
async def websocket_endpoint(ws: WebSocket, scenario_id: str) -> None:
    """
    Subscribe a browser client to the live frame stream for *scenario_id*.

    The client sends JSON command objects; the server broadcasts frame
    dicts produced by serialise_frame() via the _broadcast_loop task.

    Supported commands
    ------------------
    {"action": "pause"}    Suspend the step loop (NF-VIZ-019).
    {"action": "resume"}   Resume after pause.
    {"action": "stop"}     Destroy the scenario.
    """
    session = registry.get(scenario_id)
    if session is None:
        await ws.close(code=4004, reason="Scenario not found")
        return

    cm = session.connection_manager
    await cm.connect(ws)

    # Send world dimensions immediately so the browser can set up its view
    await ws.send_text(json.dumps({
        "type":    "hello",
        "world_x": session.world_x,
        "world_y": session.world_y,
    }))

    try:
        while True:
            text = await ws.receive_text()
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                continue
            _dispatch_command(cmd, scenario_id, session)
    except WebSocketDisconnect:
        pass
    finally:
        cm.disconnect(ws)


def _dispatch_command(cmd: dict, scenario_id: str, session: object) -> None:
    """Route a browser command to the appropriate worker control."""
    action = cmd.get("action", "")
    if action == "pause":
        session.worker.pause()
    elif action == "resume":
        session.worker.resume()
    elif action == "stop":
        registry.destroy(scenario_id)


# ---------------------------------------------------------------------------
# Broadcast background task
# ---------------------------------------------------------------------------


async def _broadcast_loop() -> None:
    """
    Drain every active session's frame queue and broadcast to clients.

    Bridges the synchronous worker queue to the async WebSocket fan-out.
    Uses get_nowait() (non-blocking) to avoid stalling the event loop
    when a session's queue is momentarily empty.

    asyncio.sleep(0) at the bottom of each iteration yields control so
    other coroutines (route handlers, WebSocket receives) remain
    responsive during high-throughput bursts.
    """
    while True:
        for session in registry.all_sessions():
            try:
                frame = session.frame_queue.get_nowait()
                await session.connection_manager.broadcast(json.dumps(frame))
            except queue.Empty:
                pass
        # Yield to the event loop every iteration
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Static files (must be last -- catches all unmatched paths)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent.parent.parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
