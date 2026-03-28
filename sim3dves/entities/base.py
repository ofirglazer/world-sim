"""
sim3dves.entities.base
======================
Abstract entity base class, type/state enumerations, and entity registry.

Design Patterns:
- Template Method: Entity.step() defines the algorithm skeleton;
  subclasses implement _update_behavior() and _update_kinematics().
- Registry: EntityManager is the authoritative index of all entities.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-004: Template Method and Registry patterns applied.
Implements: ENT-001, ENT-002, ENT-003, ENT-004.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional

import numpy as np


# ── Enumerations ───────────────────────────────────────────────────────────────

class EntityType(Enum):
    """
    Entity type catalogue (ENT-002).

    FIX vs original: raw string literals ("PEDESTRIAN") replaced with this
    enum — prevents typo bugs, enables exhaustive pattern matching.
    """
    PEDESTRIAN = auto()
    WHEELED_VEHICLE = auto()
    TRACKED_VEHICLE = auto()
    UAV = auto()


class EntityState(Enum):
    """
    Generic FSM state labels (ENT-003).

    Concrete entity subclasses may define richer state machines.
    This enum provides the minimum shared vocabulary.
    """
    IDLE = auto()
    MOVING = auto()
    ENGAGING = auto()
    DEAD = auto()


# ── Abstract Entity ────────────────────────────────────────────────────────────

class Entity(ABC):
    """
    Abstract base class for all simulated entities (ENT-001).

    Design Pattern — Template Method
    ---------------------------------
    ``step()`` defines the fixed algorithm sequence:
      1. _update_behavior()  — FSM logic, waypoint nav, social forces …
      2. _update_kinematics() — velocity integration, physics constraints
      3. _update_heading()   — derive heading from velocity vector

    Subclasses override only the two abstract hooks; the orchestration
    order and guard (``if not self.alive``) are guaranteed by the base.

    FIX vs original:
    - Was a ``@dataclass``; dataclasses with mutable numpy fields cause
      hashing, equality, and repr bugs — replaced with explicit ``__init__``.
    - ``entity_type: str`` → ``EntityType`` enum (ENT-002).
    - Added ``is_eoi`` flag (PED-004, VEH-005).
    - Added ``signature`` for detection model (VEH-006).
    - Added ``state: EntityState`` FSM field (ENT-003).
    - Added ``_update_heading()`` so all subclasses get consistent heading.
    - Added ``kill()`` helper for clean state transitions.
    """

    def __init__(
        self,
        entity_id: str,
        entity_type: EntityType,
        position: np.ndarray,
        velocity: np.ndarray,
        heading: float,
        is_eoi: bool = False,
        signature: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        entity_id : str
            Globally unique entity identifier.
        entity_type : EntityType
            Enum classifying this entity (ENT-002).
        position : np.ndarray
            3-D ENU position [x, y, z] in metres.
        velocity : np.ndarray
            3-D ENU velocity [vx, vy, vz] in m/s.
        heading : float
            Initial heading in degrees (0° = East, CCW positive).
        is_eoi : bool
            True if this entity is an Entity of Interest (PED-004, VEH-005).
        signature : float
            Optical/thermal signature factor in [0, 1] (VEH-006).
            1.0 = maximum contrast; 0.0 = invisible.
        """
        self.entity_id: str = entity_id
        self.entity_type: EntityType = entity_type
        self.position: np.ndarray = position.astype(float)
        self.velocity: np.ndarray = velocity.astype(float)
        self.heading: float = float(heading)
        self.is_eoi: bool = is_eoi
        self.signature: float = float(signature)
        self.alive: bool = True
        self.state: EntityState = EntityState.IDLE

    # ── Template Method skeleton ───────────────────────────────────

    def step(self, dt: float) -> None:
        """
        Advance entity by one simulation timestep *dt* (seconds).

        Do NOT override this method in subclasses.
        Implement ``_update_behavior()`` and ``_update_kinematics()`` instead.
        """
        if not self.alive:
            return  # Dead entities are frozen — no further updates
        self._update_behavior(dt)
        self._update_kinematics(dt)
        self._update_heading()

    # ── Abstract hooks (subclass responsibility) ───────────────────

    @abstractmethod
    def _update_behavior(self, dt: float) -> None:
        """
        Apply FSM-driven behavioral logic for one timestep.

        Examples: waypoint navigation, social force, idle transitions.
        """

    @abstractmethod
    def _update_kinematics(self, dt: float) -> None:
        """
        Integrate velocity into position and enforce physics constraints.

        Examples: Euler integration, speed clamping, terrain lock.
        """

    # ── Shared concrete helpers ────────────────────────────────────

    def _update_heading(self) -> None:
        """
        Derive heading from the XY velocity vector.

        Result is in degrees, CCW from East (ENU convention).
        Heading is unchanged if speed is negligible.
        """
        vx, vy = float(self.velocity[0]), float(self.velocity[1])
        if abs(vx) > 1e-6 or abs(vy) > 1e-6:
            self.heading = math.degrees(math.atan2(vy, vx))

    def kill(self) -> None:
        """
        Mark entity as dead and transition FSM state to DEAD.

        Use this instead of setting ``alive = False`` directly so that
        FSM state remains consistent.
        """
        self.alive = False
        self.state = EntityState.DEAD


# ── Entity Registry ────────────────────────────────────────────────────────────

class EntityManager:
    """
    Registry and batch-step coordinator for all simulation entities.

    Design Pattern — Registry
    -------------------------
    Maintains the authoritative, indexed collection of entities.
    Callers interact with the manager rather than holding direct
    references to the entity dict.

    Performance note (NF-P-001):
    The current dict-based implementation is clear and correct for M1.
    For ≥ 200 entities, replace the inner lists with NumPy structured
    arrays to enable vectorized position updates (future milestone).

    FIX vs original:
    - Added ``remove()`` for entity lifecycle management.
    - Added ``living()`` — returns only alive entities (avoids logging dead ones).
    - Added ``by_type()`` for payload FOV queries in later milestones.
    - ``add()`` now raises ``ValueError`` on duplicate IDs instead of silently
      overwriting.
    - Added ``__len__`` and ``__iter__`` for Pythonic usage.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}

    def add(self, entity: Entity) -> None:
        """
        Register *entity*. Raises ``ValueError`` on duplicate ``entity_id``.
        """
        if entity.entity_id in self._entities:
            raise ValueError(
                f"Duplicate entity_id '{entity.entity_id}' — "
                "each entity must have a unique ID."
            )
        self._entities[entity.entity_id] = entity

    def remove(self, entity_id: str) -> None:
        """Deregister entity *entity_id*. No-op if not found."""
        self._entities.pop(entity_id, None)

    def get(self, entity_id: str) -> Optional[Entity]:
        """Return entity by *entity_id*, or ``None`` if not found."""
        return self._entities.get(entity_id)

    def step_all(self, dt: float) -> None:
        """
        Call ``step(dt)`` on every entity.

        Dead entities short-circuit inside ``Entity.step()`` — no
        separate filter needed here.
        """
        for entity in self._entities.values():
            entity.step(dt)

    def all(self) -> List[Entity]:
        """Return a snapshot list of all registered entities (living + dead)."""
        return list(self._entities.values())

    def living(self) -> List[Entity]:
        """Return only entities whose ``alive`` flag is True."""
        return [e for e in self._entities.values() if e.alive]

    def by_type(self, entity_type: EntityType) -> List[Entity]:
        """
        Return living entities whose type matches *entity_type*.

        Used by the payload FOV query in later milestones.
        """
        return [
            e for e in self._entities.values()
            if e.alive and e.entity_type == entity_type
        ]

    def __iter__(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    def __len__(self) -> int:
        return len(self._entities)
