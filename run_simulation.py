"""
run_simulation.py
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

FIX vs original run_simulation.py:
-----------------------------------
1. ``np.random.rand(3) * max_x`` gave Z up to 50 m — violates Req-7.
   Replaced with ``World.snap_to_terrain()`` to lock Z to terrain elevation.
2. ``from sim3dves.viz.debug_plot import DebugPlot`` — ``viz`` sub-package
   lacked an ``__init__.py``, causing ImportError.  Fixed in package init files.
3. ``sim.step()`` called outside the logger context manager — file handle
   leaked on keyboard interrupt.  Replaced with ``with sim.logger:`` block.
4. No EOI entity designation — added 1-in-10 EOI marker for visual demo.
5. ``plot.render(sim.entities.all())`` included dead entities — now uses
   ``sim.entities.living()``.
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
from sim3dves.viz.debug_plot import DebugPlot

_D = SimDefaults()


def main() -> None:
    """Configure, populate, and run the M1 baseline scenario."""

    # ### World ###
    world = World(extent=np.array([200.0, 200.0]))

    # ### Config ###
    config = SimulationConfig(
        duration_s=30.0,
        dt=0.1,
        seed=42,
        log_file=Path("sim_log.jsonl")
    )
    sim = SimulationEngine(config, world)

    # ── Populate: terrain-snapped pedestrians ──────────────────────
    # Use a seeded RNG so spawning is also deterministic (SIM-003)
    rng = np.random.default_rng(config.seed)
    num_pedestrians = 40

    for i in range(num_pedestrians):
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

    # ### Visualiser ###
    plot = DebugPlot(world_x=world.extent[0], world_y=world.extent[1])

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
        f"  Sim time: {sim.sim_time:.1f}s"
        f"  Log: {config.log_file}"
    )


if __name__ == "__main__":
    main()
