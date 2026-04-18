"""
sim3dves.payload.cueing_policy
================================
CueingPolicy: EventBus subscriber that autonomously commands UAV orbits
and payload CUED mode when a tracked EOI reaches HIGH quality (M5).

Design Pattern: Observer — registered as a subscriber to
``EventType.TRACK_ACQUIRED`` on the engine's EventBus.  The engine fires
the event; the policy reacts without any coupling to the run loop or the
visualiser.

This is the correct home for the autonomous cueing rule that previously
lived in ``run_simulation.py``.  Moving it here achieves:
  * **Headless safety** — works identically with or without a visualiser.
  * **Testability** — the policy can be unit-tested by firing synthetic
    events, with no engine or scenario setup needed.
  * **Replaceability** — swap or subclass for different rules of
    engagement without touching the engine or the scenario.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
Implements: M5 autonomous EOI cueing (FLR-009, PAY-005).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.event_bus import Event
from sim3dves.entities.uav import UAVEntity
from sim3dves.payload.track_manager import TrackManager, TrackQuality

_D = SimDefaults()


class HighQualityEoiCueingPolicy:
    """
    Cue the primary UAV to orbit an EOI when its track reaches HIGH quality.

    Subscribes to ``EventType.TRACK_ACQUIRED``.  On each event, if the
    track is for an EOI entity and its quality is HIGH, the first living
    UAV in ``uav_entities`` is commanded to orbit the track's estimated
    position and its payload (if any) transitions to CUED mode targeting
    that entity.

    Only the first living UAV is cued (index 0).  For multi-UAV cueing
    extend this class or register additional policy instances. TODO

    Parameters
    ----------
    uav_entities : list[UAVEntity]
        Ordered list of UAV entities from the scenario.  The policy cues
        the first living entry — the scenario controls priority via order.
    track_manager : TrackManager
        Engine's track registry; used to re-fetch full track state at
        event time (the event payload carries a snapshot that may be
        slightly stale by the time the callback fires within the same
        step).
    orbit_radius_m : float
        Orbit radius to command (default: ``UAV_ORBIT_RADIUS_M``).
    orbit_altitude_m : float
        Orbit altitude to command (default: ``UAV_CRUISE_ALT_M``).

    Examples
    --------
    Register in the scenario script once; the engine fires events::

        from sim3dves.payload.cueing_policy import HighQualityEoiCueingPolicy
        from sim3dves.core.event_bus import EventType

        policy = HighQualityEoiCueingPolicy(uav_entities, sim.track_manager)
        sim.event_bus.subscribe(EventType.TRACK_ACQUIRED, policy.on_track_acquired)
    """

    def __init__(
        self,
        uav_entities: List[UAVEntity],
        track_manager: TrackManager,
        orbit_radius_m: float = _D.UAV_ORBIT_RADIUS_M,
        orbit_altitude_m: float = _D.UAV_CRUISE_ALT_M,
    ) -> None:
        self._uavs: List[UAVEntity] = uav_entities
        self._track_manager: TrackManager = track_manager
        self._orbit_radius_m: float = float(orbit_radius_m)
        self._orbit_altitude_m: float = float(orbit_altitude_m)
        # Track entity IDs that have already been cued to avoid re-cueing
        # on every TRACK_ACQUIRED event (which fires once, but guard anyway)
        self._cued_ids: set = set()

    def on_track_acquired(self, event: Event) -> None:
        """
        React to a TRACK_ACQUIRED event from the engine (M5).

        Cues the primary UAV if the event is for a HIGH-quality EOI track
        that has not already been cued this session.

        Parameters
        ----------
        event : Event
            EventBus event; ``event.payload`` contains ``entity_id``,
            ``quality``, and ``is_eoi`` keys.
        """
        entity_id: str = event.payload.get("entity_id", "")
        is_eoi: bool = bool(event.payload.get("is_eoi", False))

        if not is_eoi:
            return
        if entity_id in self._cued_ids:
            return

        # Re-fetch track for the latest estimated position
        track = self._track_manager.get_track(entity_id)
        if track is None:
            return
        if track.quality != TrackQuality.HIGH:
            return

        # Find the first living UAV to cue
        primary: Optional[UAVEntity] = next(
            (u for u in self._uavs if u.alive), None
        )
        if primary is None:
            return

        pos_xy: np.ndarray = track.position_xy
        primary.cue_orbit(
            center_xy=pos_xy,
            radius_m=self._orbit_radius_m,
            altitude_m=self._orbit_altitude_m,
        )

        payload = getattr(primary, "payload", None)
        if payload is not None:
            payload.command_cued(entity_id)

        self._cued_ids.add(entity_id)
