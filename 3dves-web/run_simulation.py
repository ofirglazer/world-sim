"""
run_simulation.py
================================
Demonstrates M2 features:
  - Wheeled vehicles: A* road navigation (VEH-001..003).
  - Tracked vehicles: off-road direct navigation (VEH-001).
  - Social-force pedestrian avoidance (PED-003).
  - Terrain-snapped spawning (Req-7, VEH-007).
  - Road network overlay in real-time visualizer (NF-VIZ-001..006).
  - Global determinism via seeded RNGs (SIM-003).
NF-CE-001..002 compliant.
=================
M3 scenario runner — demonstrates all M3 features:
  - UAVEntity with 3-D kinematics and autopilot FSM.
  - All flight rules FLR-001..011 active, including corner-escape (FLR-011).
  - NFZ cylinders with turn-rate-limited avoidance (FLR-001 fix).
  - Multi-UAV deconfliction: PRIMARY/SECONDARY orbit roles (FLR-010).
  - Cued slew: UAV-0 and UAV-1 cued to orbit EOI (FLR-009).
  - Three search patterns with corrected safe-margin waypoints (FLR-008 fix).
  - Interactive visualiser: zoom, pan, reset, entity inspection (NF-VIZ-008-015).
  - Smooth drag pan and arrow-key pan (NF-VIZ-016, NF-VIZ-017).
  - Window close stops simulation (NF-VIZ-018); pause/resume key (NF-VIZ-019).
  - All M2 features retained.
=================
M4 scenario additions:
  - OpticalPayload attached to each UAV (PAY-001..007).
  - DetectionEngine: P(D) model + vectorised LOS raycast (POL-001, NF-P-004).
  - FOV cone rendered in visualiser per UAV (NF-VIZ-006 M4).
  - DETECTION events published on EventBus and logged (PAY-004, LOG-002).
  - SimulationView used (DebugPlot alias retained for compatibility).
  - Optional logging via SimulationConfig.logging_enabled (SIM-007).
=================
M5 scenario additions:
  - TrackManager active every step; track ellipses in visualiser (NF-VIZ-006 M5).
  - HighQualityEoiCueingPolicy registered as EventBus subscriber: when an EOI
    track reaches HIGH quality, UAV-0 is autonomously cued to orbit and its
    payload transitions to CUED mode (M5, FLR-009, PAY-005).
  - SimulationRunner separates step-loop orchestration from scenario config.
    Headless: SimulationRunner(engine).run()
    Interactive: SimulationRunner(engine, visualiser=plot).run()

Architecture note
-----------------
This file is a PURE SCENARIO CONFIGURATOR.  It contains no step-loop
logic, no timing code, no render calls, and no payload-specific
orchestration.  All of that lives in:
  - SimulationRunner  (sim3dves.core.runner)      — step-loop / pacing
  - UAVEntity         (sim3dves.entities.uav)     — payload stepping
  - CueingPolicy      (sim3dves.payload.cueing_policy) — orbit cueing rule

BUG-011 fix
-----------
The HighQualityEoiCueingPolicy is now subscribed to EventType.TRACK_QUALITY_HIGH
instead of EventType.TRACK_ACQUIRED.  TRACK_QUALITY_HIGH fires at the
MEDIUM→HIGH quality promotion; TRACK_ACQUIRED continues to fire at LOW→MEDIUM.
This satisfies both test_m5.py (TRACK_ACQUIRED fires once at MEDIUM) and
test_runner.py (the policy only cues when quality is HIGH).

BUG-007 fix
-----------
SimDefaults.GRID_ORIGIN is now a plain ``Tuple[float, float]`` (immutable,
frozen-dataclass safe).  The call to RoadNetwork.build_grid() wraps it in
``np.array()`` explicitly, as that function expects an ndarray.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-M-006: All numeric constants from SimDefaults (no magic numbers).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.runner import SimulationRunner
from sim3dves.core.world import NFZCylinder, World
from sim3dves.core.event_bus import EventType
from sim3dves.entities.base import EntityType
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
from sim3dves.viz.debug_plot import SimulationView

_D = SimDefaults()
# from line_profiler import LineProfiler

# ### Scenario parameters (all sourced from _D or explicit overrides) ###
# These are scenario-level overrides, not defaults; the larger world size is
# intentional for M3 UAV coverage.  PED_NUMBER comes from SimDefaults (NF-M-006).
WORLD_X = 1500  # _D.WORLD_EXTENT_X_M  # can overide 600.0
WORLD_Y = 1500  # _D.WORLD_EXTENT_Y_M  # can overide 600.0
GRID_ROWS = 10  # _D.GRID_ROWS  # can overide 6, 6, 100.0
GRID_COLS = 10  # _D.GRID_COLS  # can overide 6, 6, 100.0
GRID_SPACING_M = _D.GRID_SPACING_M  # can overide 6, 6, 100.0
# BUG-007 fix: GRID_ORIGIN is now a tuple in SimDefaults; wrap in np.array()
# for RoadNetwork.build_grid() which expects an ndarray.
GRID_ORIGIN = np.array(_D.GRID_ORIGIN)  # np.array([50.0, 50.0])
NUM_WHEELED = _D.NUM_WHEELED  # 1  # _D.NUM_WHEELED  # can overide 12
NUM_TRACKED = _D.NUM_TRACKED  # 1  # _D.NUM_TRACKED  # can overide 5
NUM_PEDESTRIANS = _D.NUM_PEDESTRIANS  # 30  # _D.NUM_PEDESTRIANS  # can overide 40
NUM_UAVS = _D.NUM_UAVS  # 1  # _D.NUM_UAVS  # can overide 4, UAV-004: configurable multi-UAV count

# NFZ cylinders placed to exercise FLR-001 avoidance
NFZ_DEFINITIONS = []  # _D.NFZ_DEFINITIONS


def build_nfz_cylinders() -> list:
    """Construct NFZCylinder instances from the scenario table."""
    return [
        NFZCylinder(
            center_xy=np.array([cx, cy]),
            radius_m=radius,
            alt_max_m=alt_max,
        )
        for cx, cy, radius, alt_max in NFZ_DEFINITIONS
    ]


def main() -> None:
    """Configure, populate, and run the M5 scenario."""

    # ### NFZ volumes ###
    nfz_cylinders = build_nfz_cylinders()

    # ### World ###
    road_network = RoadNetwork.build_grid(
        GRID_ROWS, GRID_COLS, GRID_SPACING_M, GRID_ORIGIN,
        speed_limit_mps=_D.ROAD_SPEED_LIMIT_MPS,
    )
    world = World(
        extent=np.array([WORLD_X, WORLD_Y]),
        road_network=road_network,
        nfz_cylinders=nfz_cylinders,
        alt_floor_m=_D.UAV_ALT_FLOOR_M,
        alt_ceil_m=_D.UAV_ALT_CEIL_M,
    )

    # ### Config ###
    config = SimulationConfig(log_file=Path("../sim_log.jsonl"))
    sim = SimulationEngine(config, world)

    # ### Seeded RNG for deterministic spawning (SIM-003) ###
    rng = np.random.default_rng(config.seed)

    # ### Wheeled vehicles (VEH-001, VEH-003) ###
    node_ids = road_network.node_ids()
    kin_w = VehicleKinematics()
    for i in range(NUM_WHEELED):
        nid = node_ids[int(rng.integers(0, len(node_ids)))]
        xy = road_network.node_position(nid)
        pos = np.array([xy[0], xy[1], 0.0])  # Terrain lock (Req-7)
        sim.add_entity(WheeledVehicleEntity(
            entity_id=str(uuid.uuid4()), position=pos,
            heading=float(rng.uniform(0.0, 360.0)),
            road_network=road_network, kinematics=kin_w,
            is_eoi=(i % 5 == 0),
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        ))

    # ### Tracked vehicles (VEH-001: off-road) ###
    for i in range(NUM_TRACKED):
        xy = rng.random(2) * world.extent
        pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))  # Req-7
        sim.add_entity(TrackedVehicleEntity(
            entity_id=str(uuid.uuid4()), position=pos,
            heading=float(rng.uniform(0.0, 360.0)),
            world_extent=world.extent,
            is_eoi=(i % 5 == 1),
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        ))

    # ### Pedestrians (PED-001..003, Req-7) ###
    for i in range(NUM_PEDESTRIANS):
        xy = rng.random(2) * world.extent
        pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
        velocity = rng.standard_normal(3)
        ped = PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=pos,
            velocity=velocity,
            is_eoi=(i % 10 == 0),
        )
        sim.add_entity(ped)

    # ### UAVs (UAV-001..005, FLR-001..010) ###
    # Distribute three search patterns and one extra LAWNMOWER
    search_patterns = [
        SearchPattern.LAWNMOWER,
        SearchPattern.EXPANDING_SPIRAL,
        SearchPattern.RANDOM_WALK,
        SearchPattern.LAWNMOWER,
    ]
    uav_entities: list[UAVEntity] = []
    for i in range(NUM_UAVS):
        spawn_x = float(rng.uniform(0.0, world.extent[0]))
        spawn_y = float(rng.uniform(0.0, world.extent[1]))
        pos = np.array([spawn_x, spawn_y, _D.UAV_CRUISE_ALT_M])
        uav = UAVEntity(
            entity_id=f"uav-{i:02d}",
            position=pos,
            heading=float(rng.uniform(0.0, 360.0)),
            launch_position=pos.copy(),   # RTB target = spawn point (FLR-006)
            search_pattern=search_patterns[i % len(search_patterns)],
            cruise_altitude_m=_D.UAV_CRUISE_ALT_M + i * 10.0,  # Stagger altitudes
            endurance_s=_D.UAV_ENDURANCE_S,
            world_extent=world.extent,
            alt_floor_m=_D.UAV_ALT_FLOOR_M,
            alt_ceil_m=_D.UAV_ALT_CEIL_M,
            nfz_cylinders=nfz_cylinders,
            is_eoi=False,
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        )
        # M4: attach OpticalPayload — stepped automatically by UAVEntity (M5)
        uav.payload = OpticalPayload(
            owner_id=uav.entity_id,
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        )
        sim.add_entity(uav)
        uav_entities.append(uav)

    # ### M5: register autonomous cueing policy on the EventBus ###
    # HighQualityEoiCueingPolicy reacts to TRACK_QUALITY_HIGH events: when an
    # EOI track first reaches HIGH quality, it autonomously cues UAV-0 to orbit
    # the estimated track position and switches the payload to CUED mode.
    #
    # BUG-011 fix: subscribe to TRACK_QUALITY_HIGH (MEDIUM→HIGH transition),
    # NOT to TRACK_ACQUIRED (LOW→MEDIUM transition).  This ensures:
    #   - TRACK_ACQUIRED fires once at MEDIUM (test_m5.py contract preserved).
    #   - The policy only cues on HIGH confidence (test_runner.py contract).
    # No cueing logic lives in the step loop — the EventBus wires it here once.
    cueing_policy = HighQualityEoiCueingPolicy(
        uav_entities=uav_entities,
        track_manager=sim.track_manager,
    )
    sim.event_bus.subscribe(
        EventType.TRACK_QUALITY_HIGH, cueing_policy.on_track_acquired
    )

    print(
        f"M5: {NUM_WHEELED} wheeled | {NUM_TRACKED} tracked | "
        f"{NUM_PEDESTRIANS} peds | {NUM_UAVS} UAVs | "
        f"{len(nfz_cylinders)} NFZs | {len(road_network)} road nodes"
    )
    print("Controls: scroll=zoom | right-drag/arrows=pan | R=reset"
          " | click=select | Esc=deselect | Space=pause/resume | v=toggle world/C4I view | close=stop")

    # ### Visualiser ###
    plot = SimulationView(
        WORLD_X, WORLD_Y,
        road_network=road_network,
        nfz_cylinders=nfz_cylinders,
    )

    # ### Run — SimulationRunner owns all step-loop logic ###
    # To run headlessly: SimulationRunner(sim).run()
    runner = SimulationRunner(sim, visualiser=plot)
    runner.run()

    alive_uavs = len(sim.entities.by_type(EntityType.UAV))
    print(
        f"\nSimulation complete."
        f"  Steps: {sim.step_idx}"
        f"  Alive: {len(sim.entities.living())} (UAVs: {alive_uavs})"
        f"  Sim time: {sim.sim_time:.1f}s"
        f"  Log: {config.log_file}"
    )


if __name__ == "__main__":
    '''lp = LineProfiler()
    lp.add_function(main)
    lp.run('main()')
    lp.print_stats()'''
    main()
