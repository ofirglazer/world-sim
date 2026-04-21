"""
sim3dves.entities.base
======================
Abstract Entity base class, type/state enumerations, and EntityManager.

M2 changes
----------
* Entity.step() / _update_behavior() accept Optional[StepContext].
* EntityManager.step_all() builds per-entity StepContext via vectorised
  O(n²) neighbour search before any entity is updated (time-t consistency).

M3 changes
----------
* Entity gains ``neighbor_radius_m`` property so subclasses (UAVEntity)
  can declare a larger neighbourhood without global constant change.
* EntityManager._build_contexts() now uses per-entity radii, enabling
  UAVs to see each other at FLR-004 separation distance (50 m) while
  pedestrians retain their tight social-force radius.

Design Patterns:
Template Method (Entity), Registry (EntityManager).
NF-CE-001..005 compliant. Implements: ENT-001..004, SIM-001, NF-P-001.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.context import StepContext

_D = SimDefaults()

# ### Enumerations ###


class EntityType(Enum):
    """ Entity type catalogue — Enum, not raw strings (ENT-002)."""
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


# ### Abstract Entity ###

class Entity(ABC):
    """
    Abstract base class for all simulated entities (ENT-001).

    Design Pattern — Template Method
    ---------------------------------
    ``step()`` defines the fixed algorithm sequence:
      1. Dead-guard.
      2. _update_behavior(dt, context) — subclass hook.
      3. _update_kinematics(dt)       — subclass hook.
      4. _update_heading()            — shared concrete helper.

    Subclasses override only the two abstract hooks; the orchestration
    order and guard (``if not self.alive``) are guaranteed by the base.

    Parameters
    ----------
    entity_id : str
        Globally unique identifier.
    entity_type : EntityType
        Enum discriminator (ENT-002).
    position : np.ndarray
        3-D ENU position [x, y, z] in metres.
    velocity : np.ndarray
        3-D ENU velocity [vx, vy, vz] in m/s.
    heading : float
        Initial heading in degrees (0° = East, CCW positive).
    is_eoi : bool
        Entity of Interest flag (PED-004, VEH-005).
    signature : float
        Optical/thermal contrast in [0, 1] (VEH-006).
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
        self.entity_id: str = entity_id
        self.entity_type: EntityType = entity_type
        self.position: np.ndarray = position.astype(float)
        self.velocity: np.ndarray = velocity.astype(float)
        self.heading: float = float(heading)
        self.is_eoi: bool = is_eoi
        self.signature: float = float(signature)
        self.alive: bool = True
        self.state: EntityState = EntityState.IDLE

    # ### Template Method skeleton ###

    def step(self, dt: float, context: Optional[StepContext] = None) -> None:
        """
        Advance entity by one simulation timestep *dt* (seconds).

        Do NOT override this method in subclasses.
        Implement ``_update_behavior()`` and ``_update_kinematics()`` instead.

        Parameters
        ----------
        dt : float
            Simulation timestep (SIM-001).
        context : StepContext, optional
            Neighbourhood snapshot for social force etc. (PED-003).
        """
        if not self.alive:
            return  # Dead entities are frozen — no further updates
        self._update_behavior(dt, context)
        self._update_kinematics(dt)
        self._update_heading()

    # ### Abstract hooks (subclass responsibility) ###

    @abstractmethod
    def _update_behavior(
        self, dt: float, context: Optional[StepContext] = None
    ) -> None:
        """FSM + navigation + social force logic (subclass responsibility)."""

    @abstractmethod
    def _update_kinematics(self, dt: float) -> None:
        """Velocity integration + physics constraints (subclass responsibility)."""

    # ### Neighbourhood radius (M3: per-entity override) ###

    @property
    def neighbor_radius_m(self) -> float:
        """
        Neighbourhood search radius used by EntityManager._build_contexts().

        Subclasses override this to declare a larger or smaller search area.
        Default is NEIGHBOR_RADIUS_M (tuned for pedestrian social force).
        UAVEntity overrides to UAV_NEIGHBOR_RADIUS_M for FLR-004.

        Returns
        -------
        float
            Search radius in metres.
        """
        return _D.NEIGHBOR_RADIUS_M

    # ### Shared concrete helpers ###

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


# ### Entity Registry ###

class EntityManager:
    """
    Authoritative indexed collection and batch-step coordinator (ENT-001).

    M2: step_all() snapshots all positions into a numpy matrix, computes a
    vectorised neighbour distance matrix, builds StepContext per entity,
    then dispatches step(dt, context) — all neighbour data is from time t.

    M3: _build_contexts() uses per-entity ``neighbor_radius_m`` so UAVs
    receive a wider neighbourhood (FLR-004 separation) without increasing
    the ground-entity radius (which would slow pedestrian social force).

    Pattern: Registry. Performance: O(n²) vectorised; upgrade to spatial
    hash grid at 500+ entities (M6+). Implements: ENT-001, NF-P-001.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}

    # ### Lifecycle ###

    def add(self, entity: Entity) -> None:
        """
        Register *entity*. Raises ``ValueError`` on duplicate ``entity_id``.
        """
        if entity.entity_id in self._entities:
            raise ValueError(
                f"Duplicate entity_id '{entity.entity_id}'."
            )
        self._entities[entity.entity_id] = entity

    def remove(self, entity_id: str) -> None:
        """Deregister entity_id. No-op if not found."""
        self._entities.pop(entity_id, None)

    def get(self, entity_id: str) -> Optional[Entity]:
        """Return entity by *entity_id*, or ``None`` if not found."""
        return self._entities.get(entity_id)

    # ### Batch step ###

    def step_all(self, dt: float) -> None:
        """
        Build StepContexts then dispatch step(dt, ctx) to all living entities.

        Neighbour positions are from time t (pre-update), ensuring
        social forces are computed on a consistent state snapshot.

        Parameters
        ----------
        dt : float
            Simulation timestep (SIM-001).
        """
        living = self.living()
        if not living:
            return
        contexts = self._build_contexts(living)
        for entity in living:
            entity.step(dt, contexts[entity.entity_id])

    def _build_contexts(self, entities: List[Entity]) -> Dict[str, StepContext]:
        """
        Vectorised O(n²) neighbour search using per-entity radii (M3).

        Each entity's neighbourhood is bounded by its own
        ``neighbor_radius_m`` property, so UAVs receive a larger context
        without inflating the radius for ground entities.

        Parameters
        ----------
        entities : list[Entity]
            Snapshot of living entities (positions frozen for this step).

        Dead entities short-circuit inside ``Entity.step()`` — no
        separate filter needed here.

        Returns
        -------
        dict[str, StepContext]
            Per-entity neighbourhood snapshots.
        """
        n = len(entities)

        # (n, 2) XY position matrix for vectorised distance computation
        positions = np.array([e.position[:2] for e in entities])

        # Per-entity squared radii array — avoids repeated sqrt (M3 addition)
        radii_sq = np.array([e.neighbor_radius_m ** 2 for e in entities])

        contexts: Dict[str, StepContext] = {}
        for i, entity in enumerate(entities):
            diffs = positions - positions[i]   # (n, 2)
            # einsum "ij,ij->i" = row-wise dot product = squared distances
            sq_dists = np.einsum("ij,ij->i", diffs, diffs)  # (n,)
            # Use this entity's own radius; exclude self (sq_dist == 0)
            mask = (sq_dists < radii_sq[i]) & (sq_dists > 0.0)
            neighbours: List[Entity] = [entities[j] for j in range(n) if mask[j]]
            contexts[entity.entity_id] = StepContext(neighbors=neighbours)

        return contexts

    # ### Queries ###

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
