import numpy as np
from .base import Entity


class PedestrianEntity(Entity):
    """Minimal compliant implementation (PED-001, PED-002)."""

    def __init__(self, **kwargs):
        super().__init__(entity_type="PEDESTRIAN", **kwargs)

    def step(self, dt: float) -> None:
        noise = np.random.randn(3) * 0.1
        self.velocity += noise
        super().step(dt)
