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
Interactive scenario runner for the 3DVES M1 baseline.

Demonstrates:
- Terrain-snapped pedestrian spawning (Req-7).
- EOI entity marking (PED-004).
- Real-time matplotlib visualisation (SIM-006, M1 baseline).
- Real-time pacing via sleep budget (SIM-006).
- JSONL logging (LOG-001, LOG-002, LOG-005).

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
from sim3dves.core.world import World
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.vehicle import (
    TrackedVehicleEntity,
    VehicleKinematics,
    WheeledVehicleEntity,
)
from sim3dves.viz.debug_plot import DebugPlot
from sim3dves.maps.road_network import RoadNetwork

_D = SimDefaults()

WORLD_X, WORLD_Y = 600.0, 600.0
GRID_ROWS, GRID_COLS, GRID_SPACING_M = 6, 6, 100.0
GRID_ORIGIN = np.array([50.0, 50.0])
NUM_WHEELED, NUM_TRACKED, NUM_PEDESTRIANS = 12, 5, 40


def main() -> None:
    """Configure, populate, and run the M2 scenario."""

    # ### World ###
    road_network = RoadNetwork.build_grid(
        GRID_ROWS, GRID_COLS, GRID_SPACING_M, GRID_ORIGIN,
        speed_limit_mps=_D.ROAD_SPEED_LIMIT_MPS,
    )
    world = World(extent=np.array([WORLD_X, WORLD_Y]),  road_network=road_network)

    # ### Config ###
    config = SimulationConfig(log_file=Path("sim_log.jsonl"))
    sim = SimulationEngine(config, world)

    # ### Populate: terrain-snapped pedestrians ###
    # Use a seeded RNG so spawning is also deterministic (SIM-003)
    rng = np.random.default_rng(config.seed)
    node_ids = road_network.node_ids()

    # ### Wheeled vehicles (VEH-001, VEH-003) ###
    kin_w = VehicleKinematics()
    for i in range(NUM_WHEELED):
        nid = node_ids[int(rng.integers(0, len(node_ids)))]
        xy = road_network.node_position(nid)
        pos = np.array([xy[0], xy[1], 0.0])        # Terrain lock (Req-7)
        sim.add_entity(WheeledVehicleEntity(
            entity_id=str(uuid.uuid4()), position=pos,
            heading=float(rng.uniform(0.0, 360.0)),
            road_network=road_network, kinematics=kin_w,
            is_eoi=(i % 5 == 0),
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
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
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        ))

    # ### Pedestrians with social force (PED-001..003, Req-7) ###
    for i in range(NUM_PEDESTRIANS):
        # XY: random within world extent; Z: snapped to terrain (Req-7)
        xy = rng.random(2) * world.extent
        position = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))

        # Random initial velocity direction; speed is normalized inside entity
        velocity = rng.standard_normal(3)

        ped = PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=position,
            velocity=velocity,
            is_eoi=(i % 10 == 0),   # Mark every 10th pedestrian as EOI
        )
        sim.add_entity(ped)

    print(f"M2: {NUM_WHEELED} wheeled | {NUM_TRACKED} tracked | "
          f"{NUM_PEDESTRIANS} pedestrians | {len(road_network)} road nodes")

    # ### Visualiser ###
    plot = DebugPlot(WORLD_X, WORLD_Y, road_network=road_network)

    # ### Step loop with real-time pacing and context-managed logger ###
    steps = int(config.duration_s / config.dt)

    with sim.logger:
        for _ in range(steps):
            wall_start = time.perf_counter()
            sim.step()
            plot.render(sim.entities.living(), sim_time=sim.sim_time)

            # Real-time pacing: sleep any unused budget in this dt window
            elapsed = time.perf_counter() - wall_start
            remaining = config.dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    print(
        f"\nSimulation complete."
        f"  Steps: {sim.step_idx}"
        f"  alive: {len(sim.entities.living())}"
        f"  Sim time: {sim.sim_time:.1f}s"
        f"  Log: {config.log_file}"
    )


if __name__ == "__main__":
    main()
