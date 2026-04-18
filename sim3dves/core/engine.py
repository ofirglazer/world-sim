"""
sim3dves.core.engine
====================
Top-level simulation orchestrator.

Design Pattern: Facade — single entry point hiding sub-system complexity.
  SimulationEngine delegates to: EntityManager, EventBus, Logger, World.

M3 changes
----------
* NFZ violation detection added to step() for UAV entities: if a UAV
  position falls inside any configured NFZ after its own avoidance logic
  (FLR-001), an NFZ_VIOLATION event is published and logged (SIM-002,
  LOG-002).  The entity is NOT killed — avoidance is the UAV's responsibility.
* _handle_nfz_violation() added as EventBus subscriber.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-004: Facade pattern explicitly applied.
Implements: SIM-001 (discrete timestep), SIM-002 (event bus),
            SIM-003 (determinism), SIM-005 (headless mode),
            FLR-001 (NFZ monitoring/logging).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.event_bus import Event, EventBus, EventType
from sim3dves.core.world import World
from sim3dves.entities.base import Entity, EntityManager, EntityType
from typing import Set, Union
from sim3dves.logging.logger import Logger
from sim3dves.payload.track_manager import TrackManager, TrackState

_DEFAULTS = SimDefaults()




class _NullLogger:
    """
    No-op logger activated when SimulationConfig.logging_enabled is False (SIM-007).

    Implements identical interface to Logger so SimulationEngine call-sites
    require no conditional guards — the disabled path is transparent.
    """

    def __enter__(self) -> "_NullLogger":
        return self

    def __exit__(self, *args: object) -> bool:
        return False  # Propagate exceptions

    def log_step(self, *args: object, **kwargs: object) -> None:
        """No-op: step record suppressed when logging is disabled."""

    def log_event(self, *args: object, **kwargs: object) -> None:
        """No-op: event record suppressed when logging is disabled."""

    def close(self) -> None:
        """No-op: no file handle to close."""


@dataclass
class SimulationConfig:
    """
    Scenario-wide configuration.

    All defaults sourced from SimDefaults — NF-M-006 (no magic numbers).

    FIX vs original:
    - ``seed: int = 42`` was hardcoded inline — now sourced from SimDefaults.
    - Added ``log_file`` field so engine does not assume a fixed filename.
    """
    duration_s: float = _DEFAULTS.SIM_DURATION_S   # Total scenario length (s)
    dt: float = _DEFAULTS.SIM_DT_S                 # Timestep (s) — SIM-001
    seed: int = _DEFAULTS.SIM_SEED                 # RNG seed — SIM-003
    log_file: Path = field(
        default_factory=lambda: Path(_DEFAULTS.LOG_FILE)
    )
    logging_enabled: bool = _DEFAULTS.SIM_LOGGING_ENABLED  # SIM-007


class SimulationEngine:
    """
    Top-level Facade over all simulation sub-systems.

    Responsibilities
    ----------------
    - Advance the simulation clock one discrete timestep at a time (SIM-001).
    - Batch-step all entities via EntityManager.
    - Enforce world boundary constraints post-step.
    - Detect and log NFZ violations for UAV entities (M3, FLR-001).
    - Publish typed events on the EventBus (SIM-002).
    - Delegate structured logging to Logger (LOG-001, LOG-002, LOG-005).
    - Expose ``step_detections``: the set of entity IDs confirmed detected
      in the most recent step, for use by the visualiser (M4).
    - Advance ``track_manager``: Kalman-filter track lifecycle updated each
      step from step_detections; TRACK_ACQUIRED/TRACK_LOST events published
      on the EventBus and written to JSONL (M5).

    M5 architecture note
    --------------------
    Payload stepping is no longer performed here.  ``UAVEntity._update_behavior``
    advances its own ``payload`` component (step 7 of the UAV FSM) so the engine
    needs no payload-specific code.  Detection events are still collected here
    via ``payload.flush_detections()`` because the engine owns the EventBus.

    Determinism (SIM-003)
    ---------------------
    ``np.random.default_rng(seed)`` is used in preference to the deprecated
    ``np.random.seed()`` API.  The legacy seed is also set for any third-party
    code that still uses the global numpy RNG.

    FIX vs original:
    ----------------
    1. ``np.random.seed()`` (deprecated) → ``np.random.default_rng(seed)``.
    2. ``logger.close()`` called after the step loop — leaked file handle on
       exception.  Logger is now wrapped in ``with`` inside ``run()``.
    3. Dead entities passed to ``logger.log_step()`` — now uses ``living()``.
    4. Two full entity iterations per step (step_all + bounds check) →
       merged into a single living-entity pass in ``step()``.
    5. ``event_bus`` existed but nothing subscribed — logger now subscribes
       to OUT_OF_BOUNDS so events reach the JSONL stream (LOG-002).

    M3 FIX:
    6. NFZ violation detection added: UAV entities inside an NFZ after their
       own avoidance step trigger an NFZ_VIOLATION event (LOG-002).
    """

    def __init__(self, config: SimulationConfig, world: World) -> None:
        """
        Parameters
        ----------
        config : SimulationConfig
            Scenario parameters.
        world : World
            Immutable spatial model queried for bounds and NFZ checks.
        """
        self.config: SimulationConfig = config
        self.world: World = world

        # Deterministic RNG — preferred over deprecated np.random.seed()
        self._rng: np.random.Generator = np.random.default_rng(config.seed)
        np.random.seed(config.seed)  # Legacy seed for any third-party code

        self.event_bus: EventBus = EventBus()
        self.entities: EntityManager = EntityManager()

        # Logger is created here; the context manager is applied in run()
        # so interactive (non-run) use can still call step() manually.
        # SIM-007: when logging_enabled=False use a no-op _NullLogger so
        # all call-sites remain identical — no conditional guards needed.
        self.logger: Union[Logger, _NullLogger] = (
            Logger(config.log_file)
            if config.logging_enabled
            else _NullLogger()
        )

        # Wire event handlers — all boundary and NFZ events reach JSONL
        # Wire logger to event bus — all events reach the JSONL stream (LOG-002)
        self.event_bus.subscribe(
            EventType.OUT_OF_BOUNDS, self._handle_out_of_bounds
        )
        # M3: wire NFZ_VIOLATION events to logger (LOG-002, FLR-001)
        self.event_bus.subscribe(
            EventType.NFZ_VIOLATION, self._handle_nfz_violation
        )
        # M4: wire DETECTION events to logger (LOG-002, PAY-004)
        self.event_bus.subscribe(
            EventType.DETECTION, self._handle_detection
        )
        # M5: wire track lifecycle events to logger (LOG-002)
        self.event_bus.subscribe(
            EventType.TRACK_ACQUIRED, self._handle_track_event
        )
        self.event_bus.subscribe(
            EventType.TRACK_LOST, self._handle_track_event
        )

        self.sim_time: float = 0.0   # Current simulation time (s)
        self.step_idx: int = 0       # Zero-based step counter
        # Entity IDs confirmed detected in the most recent step (M4).
        # Cleared at the top of each step; populated during flush_detections.
        # Passed to render() so the visualiser can flash a detection ring.
        self.step_detections: Set[str] = set()

        # TrackManager: Kalman-filter track registry updated each step (M5).
        # Callbacks publish TRACK_ACQUIRED / TRACK_LOST events on the bus.
        self.track_manager: TrackManager = TrackManager(
            on_track_acquired=self._on_track_acquired,
            on_track_lost=self._on_track_lost,
        )

    # ### Public API ###

    def add_entity(self, entity: Entity) -> None:
        """Register *entity* with the simulation before the first step."""
        self.entities.add(entity)

    def step(self) -> float:
        """
        Advance simulation by one timestep ``config.dt``.

        Executes in a single pass over living entities:
          1. Behavioral + kinematic update (EntityManager.step_all).
          2. World boundary check → kill + publish event if violated.
          3. NFZ violation check for UAVs → publish event (M3, FLR-001).
          4. Log step record.

        Returns
        -------
        float
            Wall-clock seconds consumed by this step (LOG-005).
        """
        wall_start = time.perf_counter()
        dt = self.config.dt

        # 1. Advance all living entities (dead entities short-circuit inside step())
        self.entities.step_all(dt)

        # 2. Single-pass boundary enforcement on the now-updated positions
        for entity in self.entities.living():
            if not self.world.in_bounds(entity.position):
                entity.kill()
                self.event_bus.publish(Event(
                    timestamp=self.sim_time,
                    event_type=EventType.OUT_OF_BOUNDS,
                    payload={
                        "id": entity.entity_id,
                        "pos": entity.position.tolist(),
                    },
                ))

        # 3. NFZ violation monitoring for UAVs (M3, FLR-001, LOG-002).
        #    The UAV's own avoidance logic (FLR-001) should prevent entry;
        #    this check logs residual penetration AND keeps a per-entity
        #    persistent flag (_nfz_violated_flag) so the inspection panel
        #    reflects current violation state every step (Bug 2 fix).
        for entity in self.entities.by_type(EntityType.UAV):
            # Use both the world's NFZ list and the entity's own list so
            # the flag is accurate regardless of which source is populated.
            violated = self.world.in_nfz(entity.position) or bool(
                getattr(entity, "nfz_violated", False)
            )
            # Write back a plain bool attribute so the panel can read it
            # without calling the property (which needs _nfz_cylinders set).
            entity._nfz_violated_flag = violated
            if violated:
                self.event_bus.publish(Event(
                    timestamp=self.sim_time,
                    event_type=EventType.NFZ_VIOLATION,
                    payload={
                        "id": entity.entity_id,
                        "pos": entity.position.tolist(),
                    },
                ))

        # 4. Collect and publish DETECTION events from UAV payloads (M4, PAY-004).
        # step_detections is reset here so it only ever contains IDs from
        # the *current* step — the visualiser treats it as a one-frame flash.
        self.step_detections = set()
        for entity in self.entities.by_type(EntityType.UAV):
            payload = getattr(entity, "payload", None)
            if payload is None:
                continue
            for det in payload.flush_detections():
                self.step_detections.add(det["target_id"])  # accumulate for viz
                self.event_bus.publish(Event(
                    timestamp=self.sim_time,
                    event_type=EventType.DETECTION,
                    payload=det,
                ))

        # 5. Advance TrackManager: Kalman predict + update from detections (M5)
        living_for_track = self.entities.living()
        self.track_manager.step(
            detected_ids=self.step_detections,
            entities=living_for_track,
            dt=dt,
        )

        # 6. Log living entities only — dead entities add noise, not value
        wall_dt = time.perf_counter() - wall_start
        self.logger.log_step(
            self.step_idx,
            self.entities.living(),
            wall_dt_s=wall_dt,
        )

        # 5. Advance clocks
        self.sim_time += dt
        self.step_idx += 1

        return wall_dt

    def run(self) -> None:
        """
        Execute the full scenario duration headlessly (SIM-005).

        Prefer ``SimulationRunner`` (``sim3dves.core.runner``) for both
        headless and interactive use — it handles real-time pacing, pause,
        and window-close cleanly.  This method is retained for backwards
        compatibility and for minimal headless scripts that do not need
        real-time pacing.

        The Logger is used as a context manager here to guarantee the
        JSONL file is flushed and closed even if an exception escapes
        the step loop.
        """
        steps = int(self.config.duration_s / self.config.dt)
        with self.logger:
            for _ in range(steps):
                self.step()

    # ### Private event handlers ###

    def _handle_out_of_bounds(self, event: Event) -> None:
        """
        Write an OUT_OF_BOUNDS event record to the JSONL stream (LOG-002).

        Subscribed to EventBus in __init__ so all boundary violations
        are captured automatically.
        """
        self.logger.log_event({
            "type": EventType.OUT_OF_BOUNDS.name,
            "timestamp": event.timestamp,
            **event.payload,
        })

    def _handle_nfz_violation(self, event: Event) -> None:
        """
        Write an NFZ_VIOLATION event record to the JSONL stream (LOG-002, FLR-001).

        Subscribed to EventBus in __init__.  A violation means a UAV
        penetrated an NFZ despite the avoidance logic — indicates a
        scenario where the avoidance was too slow to react.
        """
        self.logger.log_event({
            "type": EventType.NFZ_VIOLATION.name,
            "timestamp": event.timestamp,
            **event.payload,
        })

    def _handle_detection(self, event: Event) -> None:
        """
        Write a DETECTION event record to the JSONL stream (LOG-002, PAY-004).

        Subscribed to EventBus in __init__.  Published each step for every
        entity detected inside any UAV payload FOV with P(D) > threshold.
        """
        self.logger.log_event({
            "type": EventType.DETECTION.name,
            "timestamp": event.timestamp,
            **event.payload,
        })

    def _on_track_acquired(self, entity_id: str, track: TrackState) -> None:
        """
        Callback from TrackManager when a track reaches MEDIUM quality (M5).

        Publishes TRACK_ACQUIRED on the EventBus so the logger and any other
        subscribers (e.g. a future ROE module) receive the event.
        """
        self.event_bus.publish(Event(
            timestamp=self.sim_time,
            event_type=EventType.TRACK_ACQUIRED,
            payload={
                "entity_id": entity_id,
                "quality":   track.quality.name,
                "is_eoi":    track.is_eoi,
                "pos":       track.position_xy.tolist(),
            },
        ))

    def _on_track_lost(self, entity_id: str, track: TrackState) -> None:
        """
        Callback from TrackManager when a track is removed (M5).

        Publishes TRACK_LOST on the EventBus.
        """
        self.event_bus.publish(Event(
            timestamp=self.sim_time,
            event_type=EventType.TRACK_LOST,
            payload={
                "entity_id":  entity_id,
                "age_steps":  track.age_steps,
                "miss_count": track.miss_count,
                "pos":        track.position_xy.tolist(),
            },
        ))

    def _handle_track_event(self, event: Event) -> None:
        """
        Write TRACK_ACQUIRED / TRACK_LOST records to JSONL (LOG-002, M5).

        Subscribed to both EventType.TRACK_ACQUIRED and EventType.TRACK_LOST.
        """
        self.logger.log_event({
            "type":      event.event_type.name,
            "timestamp": event.timestamp,
            **event.payload,
        })
