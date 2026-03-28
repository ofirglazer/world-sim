import numpy as np
import uuid

from sim3dves.core.engine import SimulationEngine, SimulationConfig
from sim3dves.core.world import World
from sim3dves.entities.pedestrian import PedestrianEntity


if __name__ == "__main__":
    config = SimulationConfig(duration_s=2.0, dt=0.1)
    world = World(extent=np.array([1000.0, 1000.0]))
    sim = SimulationEngine(config, world)

    for _ in range(5):
        sim.add_entity(PedestrianEntity(
            entity_id=str(uuid.uuid4()),
            position=np.random.rand(3) * 100,
            velocity=np.random.randn(3),
            heading=0.0,
        ))

    sim.run()
