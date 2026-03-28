import json
from typing import List
from sim3dves.entities.base import Entity


class Logger:
    """Production-ready logger skeleton (expand to events + performance)."""

    def __init__(self, filepath: str):
        self.file = open(filepath, "w")

    def log_step(self, step: int, entities: List[Entity]) -> None:
        self.file.write(json.dumps({
            "step": step,
            "entities": [
                {
                    "id": e.entity_id,
                    "type": e.entity_type,
                    "pos": e.position.tolist(),
                    "vel": e.velocity.tolist(),
                }
                for e in entities
            ],
        }) + "\n")

    def close(self):
        self.file.close()
