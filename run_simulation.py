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
Demonstrates M3 features:
  - UAVEntity: 3-D kinematics, autopilot FSM (WAYPOINT/LOITER/ORBIT/RTB).
  - All flight rules FLR-001..010 active.
  - NFZ cylinders: avoidance and JSONL violation logging.
  - Multi-UAV deconfliction: PRIMARY/SECONDARY orbit roles (FLR-010).
  - Cued slew: one UAV transitioned to ORBIT around EOI (FLR-009).
  - Three search patterns: LAWNMOWER, EXPANDING_SPIRAL, RANDOM_WALK.
  - NFZ circles and geofence boundary in visualiser (NF-VIZ-006 M3).
  - All M2 features retained: wheeled/tracked vehicles, social force,
    road network, real-time pacing.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.world import NFZCylinder, World
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.uav import AutopilotMode, SearchPattern, UAVEntity
from sim3dves.entities.vehicle import (
    TrackedVehicleEntity,
    VehicleKinematics,
    WheeledVehicleEntity,
)
from sim3dves.viz.debug_plot import DebugPlot
from sim3dves.maps.road_network import RoadNetwork

# from line_profiler import LineProfiler

_D = SimDefaults()

# ### Scenario constants ###
WORLD_X, WORLD_Y = 600.0, 600.0
GRID_ROWS, GRID_COLS, GRID_SPACING_M = 6, 6, 100.0
GRID_ORIGIN = np.array([50.0, 50.0])
NUM_WHEELED, NUM_TRACKED, NUM_PEDESTRIANS = 12, 5, 40
NUM_UAVS = 4   # UAV-004: configurable multi-UAV count

# NFZ cylinders placed to exercise FLR-001 avoidance
NFZ_DEFINITIONS = [
    # (centre_x, centre_y, radius_m, alt_max_m)
    (200.0, 300.0, 60.0, 200.0),
    (450.0, 150.0, 50.0, 150.0),
    (400.0, 450.0, 70.0, 300.0),
]

# Step at which UAV-0 is cued to orbit the first EOI pedestrian (FLR-009)
CUE_ORBIT_STEP: int = 30


def build_nfz_cylinders() -> list:
    """Construct NFZCylinder instances from the scenario definition table."""
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
    node_ids = road_network.node_ids()

    # ### Wheeled vehicles (VEH-001, VEH-003) ###
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
    eoi_ped_pos: np.ndarray | None = None   # Track first EOI pedestrian for cue
    for i in range(NUM_PEDESTRIANS):
        # XY: random within world extent; Z: snapped to terrain (Req-7)
        xy = rng.random(2) * world.extent
        position = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))

        # Random initial velocity direction; speed is normalized inside entity
        velocity = rng.standard_normal(3)
        is_eoi = (i % 10 == 0)
        ped = PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=position,
            velocity=velocity,
            is_eoi=is_eoi,
        )
        sim.add_entity(ped)
        if is_eoi and eoi_ped_pos is None:
            eoi_ped_pos = position.copy()   # Capture for FLR-009 demo

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
        # Spawn at different quadrant centres at cruise altitude
        spawn_x = float(rng.uniform(100.0, 500.0))
        spawn_y = float(rng.uniform(100.0, 500.0))
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

    print(
        f"M3: {NUM_WHEELED} wheeled | {NUM_TRACKED} tracked | "
        f"{NUM_PEDESTRIANS} peds | {NUM_UAVS} UAVs | "
        f"{len(nfz_cylinders)} NFZs | {len(road_network)} road nodes"
    )

    # ### Visualiser (with NFZ circles and geofence boundary, M3) ###
    plot = DebugPlot(
        WORLD_X, WORLD_Y,
        road_network=road_network,
        nfz_cylinders=nfz_cylinders,
    )

    # ### Step loop ###
    steps = int(config.duration_s / config.dt)

    with sim.logger:
        for step in range(steps):
            wall_start = time.perf_counter()
            sim.step()

            # FLR-009 demo: cue UAV-0 to orbit the first EOI pedestrian
            if step == CUE_ORBIT_STEP and eoi_ped_pos is not None and uav_entities:
                uav_entities[0].cue_orbit(
                    center_xy=eoi_ped_pos[:2],
                    radius_m=_D.UAV_ORBIT_RADIUS_M,
                    altitude_m=_D.UAV_CRUISE_ALT_M,
                )
                # Cue UAV-1 to same point to trigger FLR-010 deconfliction
                if len(uav_entities) > 1:
                    uav_entities[1].cue_orbit(
                        center_xy=eoi_ped_pos[:2],
                        radius_m=_D.UAV_ORBIT_RADIUS_M,
                        altitude_m=_D.UAV_CRUISE_ALT_M,
                    )
                print(f"  Step {step}: UAV-0 and UAV-1 cued to orbit EOI at "
                      f"({eoi_ped_pos[0]:.0f}, {eoi_ped_pos[1]:.0f})")

            plot.render(sim.entities.living(), sim_time=sim.sim_time)

            # Real-time pacing: sleep unused dt budget (SIM-006)
            elapsed = time.perf_counter() - wall_start
            remaining = config.dt - elapsed
            print(remaining)
            if remaining > 0.0:
                time.sleep(remaining)

    alive = len(sim.entities.living())
    uavs_alive = len(sim.entities.by_type(
        __import__("sim3dves.entities.base", fromlist=["EntityType"]).EntityType.UAV
    ))
    print(
        f"\nSimulation complete."
        f"  Steps: {sim.step_idx}"
        f"  Alive: {alive} (UAVs: {uavs_alive})"
        f"  Sim time: {sim.sim_time:.1f}s"
        f"  Log: {config.log_file}"
    )


if __name__ == "__main__":
    '''lp = LineProfiler()
    lp.add_function(main)
    lp.run('main()')
    lp.print_stats()'''
    main()
