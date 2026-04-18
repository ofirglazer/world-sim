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
  - Smooth drag pan, arrow-key pan (NF-VIZ-016, NF-VIZ-017).
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
  - Autonomous EOI cueing: when a track reaches HIGH quality and is EOI,
    UAV-0's payload transitions to CUED mode and UAV-0 orbits the track (M5).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-M-006: All numeric constants from SimDefaults (no magic numbers).
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.world import NFZCylinder, World
from sim3dves.entities.base import EntityType
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.uav import SearchPattern, UAVEntity
from sim3dves.entities.vehicle import (
    TrackedVehicleEntity,
    VehicleKinematics,
    WheeledVehicleEntity,
)
from sim3dves.maps.road_network import RoadNetwork
from sim3dves.payload.optical_payload import GimbalMode, OpticalPayload
from sim3dves.viz.debug_plot import DebugPlot, SimulationView

# from line_profiler import LineProfiler

_D = SimDefaults()

# ### Scenario parameters (all sourced from _D or explicit overrides) ###
# These are scenario-level overrides, not defaults; the larger world size is
# intentional for M3 UAV coverage.  PED_NUMBER comes from SimDefaults (NF-M-006).
WORLD_X = 1500  # _D.WORLD_EXTENT_X_M  # can overide 600.0
WORLD_Y = 1500  # _D.WORLD_EXTENT_Y_M  # can overide 600.0
GRID_ROWS = 10  # _D.GRID_ROWS  # can overide 6, 6, 100.0
GRID_COLS = 10  # _D.GRID_COLS  # can overide 6, 6, 100.0
GRID_SPACING_M = _D.GRID_SPACING_M  # can overide 6, 6, 100.0
GRID_ORIGIN = _D.GRID_ORIGIN  # can overide np.array([50.0, 50.0])
NUM_WHEELED = _D.NUM_WHEELED  # 1  # _D.NUM_WHEELED  # can overide 12
NUM_TRACKED = _D.NUM_TRACKED  # 1  # _D.NUM_TRACKED  # can overide 5
NUM_PEDESTRIANS = _D.NUM_PEDESTRIANS  #30  # _D.NUM_PEDESTRIANS  # can overide 40
NUM_UAVS = _D.NUM_UAVS  # 1  # _D.NUM_UAVS  # can overide 4, UAV-004: configurable multi-UAV count

# NFZ cylinders placed to exercise FLR-001 avoidance
NFZ_DEFINITIONS = []  # _D.NFZ_DEFINITIONS

# Step at which UAV-0/1 are cued to orbit the first EOI pedestrian (FLR-009)
CUE_ORBIT_STEP: int = 30


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
    """Configure, populate, and run the M3 scenario."""

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
    config = SimulationConfig(log_file=Path("sim_log.jsonl"))
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
    eoi_ped_pos: np.ndarray | None = None
    for i in range(NUM_PEDESTRIANS):
        # XY: random within world extent; Z: snapped to terrain (Req-7)
        xy = rng.random(2) * world.extent
        pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))

        # Random initial velocity direction; speed is normalized inside entity
        velocity = rng.standard_normal(3)
        is_eoi = (i % 10 == 0)
        ped = PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=pos,
            velocity=velocity,
            is_eoi=is_eoi,
        )
        sim.add_entity(ped)
        if is_eoi and eoi_ped_pos is None:
            eoi_ped_pos = pos.copy()   # Capture for FLR-009 demo

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
        # Spawn UAVs at cruise altitude
        spawn_x = float(rng.uniform(0.0, world.extent[0]))  # consider 200.0, 400.0 for spawn
        spawn_y = float(rng.uniform(0.0, world.extent[1]))  # consider 200.0, 400.0 for spawn
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
        sim.add_entity(uav)
        uav_entities.append(uav)
        # M4: attach one OpticalPayload per UAV (PAY-001, FLR-009).
        # Pass a seeded RNG so Bernoulli detection draws are deterministic
        # and reproducible given the same scenario seed (SIM-003).
        uav.payload = OpticalPayload(
            owner_id=uav.entity_id,
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
        )

    print(
        f"M5: {NUM_WHEELED} wheeled | {NUM_TRACKED} tracked | "
        f"{NUM_PEDESTRIANS} peds | {NUM_UAVS} UAVs | "
        f"{len(nfz_cylinders)} NFZs | {len(road_network)} road nodes"
    )
    print("Controls: scroll=zoom | right-drag/arrows=pan | R=reset"
          " | click=select | Esc=deselect | Space=pause/resume | v=toggle world/C4I view | close=stop")

    # ### Visualiser (M3 interactive, NF-VIZ-008..015) ###
    # M4: use SimulationView (DebugPlot alias) with FOV cone support
    plot = SimulationView(
        WORLD_X, WORLD_Y,
        road_network=road_network,
        nfz_cylinders=nfz_cylinders,
    )

    # ### Step loop ###
    steps = int(config.duration_s / config.dt)

    with sim.logger:
        # NF-VIZ-018/019: use a while loop so the step budget is only consumed
        # when a real simulation step executes.  A 'continue' in the pause
        # branch of a for-loop would silently advance the loop counter even
        # though sim.step() was never called (Bug 1 fix).
        step = 0
        while step < steps:
            # NF-VIZ-018: window close → break cleanly; logger flushes on exit
            if plot.window_closed:
                break

            # NF-VIZ-019: pause → render without stepping; budget unchanged
            if plot.paused:
                plot.render(
                    sim.entities.living(),
                    sim_time=sim.sim_time,
                    track_manager=sim.track_manager,
                )
                time.sleep(config.dt)
                continue  # step NOT incremented — budget is preserved

            wall_start = time.perf_counter()  # wall means the real time elapsed
            elapsed_step = sim.step()
            print(f"Elapsed time in sim.step: {elapsed_step:.4f} sec")

            # M4: step each UAV payload (PAY-001..007, NF-P-004)
            living_entities = sim.entities.living()
            for uav in uav_entities:
                if uav.alive and hasattr(uav, "payload"):
                    uav.payload.step(
                        uav_position=uav.position,
                        uav_heading_deg=uav.heading,
                        entities=living_entities,
                        dt=config.dt,
                        structures=world.structures,
                    )

            # M5: autonomous EOI cueing via TrackManager.
            # When any EOI track reaches HIGH quality, command UAV-0 to orbit
            # and its payload to CUED mode targeting the tracked entity.
            for _eoi_track in sim.track_manager.eoi_tracks():
                from sim3dves.payload.track_manager import TrackQuality
                if (_eoi_track.quality == TrackQuality.HIGH
                        and len(uav_entities) > 0
                        and uav_entities[0].alive):
                    _uav0 = uav_entities[0]
                    _pos = _eoi_track.position_xy
                    _uav0.cue_orbit(
                        center_xy=_pos,
                        radius_m=_D.UAV_ORBIT_RADIUS_M,
                        altitude_m=_D.UAV_CRUISE_ALT_M,
                    )
                    if hasattr(_uav0, "payload"):
                        _uav0.payload.command_cued(_eoi_track.entity_id)

            # FLR-009: cue two UAVs to orbit the first EOI at step 30;
            # also command their payloads to CUED mode (PAY-005)
            if step == CUE_ORBIT_STEP and eoi_ped_pos is not None:
                for idx in range(min(2, len(uav_entities))):
                    uav_entities[idx].cue_orbit(
                        center_xy=eoi_ped_pos[:2],
                        radius_m=_D.UAV_ORBIT_RADIUS_M,
                        altitude_m=_D.UAV_CRUISE_ALT_M,
                    )
                    if hasattr(uav_entities[idx], "payload"):
                        uav_entities[idx].payload.command_stare(eoi_ped_pos[:2])
                print(
                    f"  Step {step}: UAV-0/1 cued to orbit EOI at "
                    f"({eoi_ped_pos[0]:.0f}, {eoi_ped_pos[1]:.0f})"
                )

            # Pass step_detections and track_manager so the visualiser can
            # flash detection rings (M4) and draw track ellipses (M5).
            plot.render(
                sim.entities.living(),
                sim_time=sim.sim_time,
                detected_ids=sim.step_detections,
                track_manager=sim.track_manager,
            )

            # Real-time pacing: sleep unused dt budget (SIM-006)
            elapsed = time.perf_counter() - wall_start
            remaining = config.dt - elapsed
            print(f"remaining time is {remaining}")
            if remaining > 0.0:
                time.sleep(remaining)
            step += 1  # only reached when sim.step() actually ran

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
