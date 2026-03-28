from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class Entity:
    entity_id: str
    entity_type: str
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    alive: bool = True

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt


class EntityManager:
    """Future optimization: vectorized storage."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}

    def add(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def step_all(self, dt: float) -> None:
        for e in self.entities.values():
            if e.alive:
                e.step(dt)

    def all(self) -> List[Entity]:
        return list(self.entities.values())
