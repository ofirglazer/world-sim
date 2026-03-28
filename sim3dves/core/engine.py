from dataclasses import dataclass
import numpy as np

from sim3dves.core.event_bus import EventBus, Event
from sim3dves.core.world import World
from sim3dves.entities.base import EntityManager, Entity
from sim3dves.logging.logger import Logger


@dataclass
class SimulationConfig:
    duration_s: float
    dt: float
    seed: int = 42


class SimulationEngine:
    """Deterministic simulation core."""

    def __init__(self, config: SimulationConfig, world: World):
        self.config = config
        self.world = world
        self.event_bus = EventBus()
        self.entities = EntityManager()
        self.logger = Logger("sim_log.jsonl")

        self.time = 0.0
        self.step_idx = 0

        np.random.seed(config.seed)

    def add_entity(self, entity: Entity) -> None:
        self.entities.add(entity)

    def step(self) -> None:
        dt = self.config.dt

        self.entities.step_all(dt)

        # Enforce world constraints
        for e in self.entities.all():
            if not self.world.in_bounds(e.position):
                e.alive = False
                self.event_bus.publish(Event(self.time, "OUT_OF_BOUNDS", {"id": e.entity_id}))

        self.logger.log_step(self.step_idx, self.entities.all())

        self.time += dt
        self.step_idx += 1

    def run(self) -> None:
        steps = int(self.config.duration_s / self.config.dt)
        for _ in range(steps):
            self.step()
        self.logger.close()
