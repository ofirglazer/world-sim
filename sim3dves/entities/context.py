"""
sim3dves.entities.context
=========================
StepContext — lightweight per-entity neighbourhood snapshot.

Built once per simulation step by EntityManager._build_contexts()
*before* any entity is updated, so all neighbour positions are from
time t rather than a partially-updated t+dt.

Design note: StepContext intentionally uses ``List[Any]`` for the
neighbours field to avoid a circular import with
sim3dves.entities.base (which defines Entity).  Consumers cast to
``Entity`` via their own import.  This is the standard Python
pattern for intra-package forward references.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-004: Data-Transfer Object pattern.
Implements: PED-003 (social-force support), ENT-001 (context awareness).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class StepContext:
    """
    Snapshot of an entity's neighbourhood at the start of one step.

    Attributes
    ----------
    neighbors : list
        Living entities whose XY position is within the configured
        neighbour-search radius of the owner entity.
        Runtime type: ``List[sim3dves.entities.base.Entity]``.
        Empty list when no neighbours are within range.
    """

    # Runtime type is List[Entity]; using Any to break the circular import.
    neighbors: List[Any] = field(default_factory=list)
