from dataclasses import dataclass
import numpy as np


@dataclass
class World:
    extent: np.ndarray

    def in_bounds(self, position: np.ndarray) -> bool:
        return (
            0 <= position[0] <= self.extent[0]
            and 0 <= position[1] <= self.extent[1]
        )
