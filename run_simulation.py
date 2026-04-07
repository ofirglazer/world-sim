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
  - All M2 features retained.

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
from sim3dves.viz.debug_plot import DebugPlot

# from line_profiler import LineProfiler

_D = SimDefaults()

# ### Scenario parameters (all sourced from _D or explicit overrides) ###
# These are scenario-level overrides, not defaults; the larger world size is
# intentional for M3 UAV coverage.  PED_NUMBER comes from SimDefaults (NF-M-006).
WORLD_X = 2000  # _D.WORLD_EXTENT_X_M  # can overide 600.0
WORLD_Y = 2000  # _D.WORLD_EXTENT_Y_M  # can overide 600.0
GRID_ROWS, GRID_COLS, GRID_SPACING_M = _D.GRID_ROWS, _D.GRID_COLS, _D.GRID_SPACING_M  # can overide 6, 6, 100.0
GRID_ORIGIN = _D.GRID_ORIGIN  # can overide np.array([50.0, 50.0])
NUM_WHEELED = 1  # _D.NUM_WHEELED  # can overide 12
NUM_TRACKED = 1  # _D.NUM_TRACKED  # can overide 5
NUM_PEDESTRIANS = 1  # _D.NUM_PEDESTRIANS  # can overide 40
NUM_UAVS = 8  # _D.NUM_UAVS  # can overide 4, UAV-004: configurable multi-UAV count

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

    print(
        f"M3: {NUM_WHEELED} wheeled | {NUM_TRACKED} tracked | "
        f"{NUM_PEDESTRIANS} peds | {NUM_UAVS} UAVs | "
        f"{len(nfz_cylinders)} NFZs | {len(road_network)} road nodes"
    )
    print("Controls: scroll=zoom | drag=pan | R=reset | click=select | Esc=deselect")

    # ### Visualiser (M3 interactive, NF-VIZ-008..015) ###
    plot = DebugPlot(
        WORLD_X, WORLD_Y,
        road_network=road_network,
        nfz_cylinders=nfz_cylinders,
    )

    # ### Step loop ###
    steps = int(config.duration_s / config.dt)

    with sim.logger:
        for step in range(steps):
            wall_start = time.perf_counter()  # wall means the real tine elapsed time
            elapsed_step = sim.step()
            # print(f"Elapsed time in sim.step: {elapsed_step:.4f} sec")

            # FLR-009: cue two UAVs to orbit the first EOI at step 30
            if step == CUE_ORBIT_STEP and eoi_ped_pos is not None:
                for idx in range(min(2, len(uav_entities))):
                    uav_entities[idx].cue_orbit(
                        center_xy=eoi_ped_pos[:2],
                        radius_m=_D.UAV_ORBIT_RADIUS_M,
                        altitude_m=_D.UAV_CRUISE_ALT_M,
                    )
                print(
                    f"  Step {step}: UAV-0/1 cued to orbit EOI at "
                    f"({eoi_ped_pos[0]:.0f}, {eoi_ped_pos[1]:.0f})"
                )

            plot.render(sim.entities.living(), sim_time=sim.sim_time)

            # Real-time pacing: sleep unused dt budget (SIM-006)
            elapsed = time.perf_counter() - wall_start
            remaining = config.dt - elapsed
            # print(remaining)
            if remaining > 0.0:
                time.sleep(remaining)

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
