import numpy as np
import uuid
import time

from sim3dves.core.engine import SimulationEngine, SimulationConfig
from sim3dves.core.world import World
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.viz.debug_plot import DebugPlot  # your new module


def main():
    max_x = 50.0
    max_y = 50.0
    config = SimulationConfig(duration_s=10.0, dt=0.1)
    world = World(extent=np.array([max_x, max_y]))
    sim = SimulationEngine(config, world)

    # Add entities
    for _ in range(20):
        sim.add_entity(PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=np.random.rand(3) * max_x,
            velocity=np.random.randn(3),
            heading=0.0,
        ))

    # Create visualizer
    plot = DebugPlot(max_x, max_y)

    steps = int(config.duration_s / config.dt)

    for _ in range(steps):
        start = time.time()

        sim.step()

        # 👇 THIS is the hook
        plot.render(sim.entities.all())

        # Optional real-time pacing
        elapsed = time.time() - start
        sleep_time = config.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
