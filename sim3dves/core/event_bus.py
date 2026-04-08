"""
sim3dves.core.event_bus
=======================
Synchronous publish-subscribe event bus.

Design Pattern: Observer — decouples publishers from subscribers.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-004: Observer pattern explicitly applied.
Implements: SIM-002 (event bus).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List


class EventType(Enum):
    """
    Typed event catalogue.

    FIX vs original: raw string literals ("OUT_OF_BOUNDS") replaced with
    this enum — prevents typo bugs and enables IDE completion.
    """
    OUT_OF_BOUNDS = auto()
    ENTITY_DIED = auto()
    STEP_COMPLETE = auto()
    DETECTION = auto()
    TRACK_ACQUIRED = auto()
    TRACK_LOST = auto()
    NFZ_VIOLATION = auto()
    LOW_FUEL = auto()


@dataclass
class Event:
    """
    Immutable event record published on the bus.
    """
    timestamp: float  # Simulation time (s) of the event
    event_type: EventType  # Typed event discriminator
    payload: Dict[str, Any] = field(default_factory=dict)  # Arbitrary metadata


# Type alias for subscriber callables
Subscriber = Callable[[Event], None]


class EventBus:
    """
    Synchronous publish-subscribe event dispatcher.

    Thread-safety note: This implementation is intentionally single-threaded
    for M1. The comment in the original code claiming thread-safety was
    incorrect — there are no locks. To extend for multithreaded use,
    wrap ``_subscribers`` access with ``threading.Lock``.

    FIX vs original:
    - ``event.type: str`` → ``event.event_type: EventType``
    - Added ``unsubscribe()`` for lifecycle management
    - Removed misleading thread-safety comment
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Subscriber]] = {}

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Register *handler* to be called whenever *event_type* is published."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Remove a previously registered *handler* for *event_type*."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Dispatch *event* synchronously to all registered subscribers."""
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)
