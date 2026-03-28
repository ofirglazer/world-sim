from dataclasses import dataclass
from typing import Dict, List, Callable, Any


@dataclass
class Event:
    timestamp: float
    type: str
    payload: Dict[str, Any]


class EventBus:
    """Thread-safe ready pub-sub (extensible to priority queue later)."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers.get(event.type, []):
            handler(event)
