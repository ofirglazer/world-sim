"""
sim3dves.core.runner
====================
SimulationRunner: separates the step-loop orchestration from scenario
configuration and from the engine itself.

Design Pattern: Strategy — the runner accepts an optional visualiser;
when None, it runs headlessly.  All visualiser-specific code (pause,
window-close polling, render calls) is isolated here and never leaks
into the engine or scenario scripts.

Motivation
----------
Previously the step-loop, pause handling, window-close detection, and
render calls all lived inside ``run_simulation.py``.  This made headless
execution impossible without modifying the scenario script, and it mixed
scenario policy (cueing) with infrastructure (timing, rendering).

``SimulationRunner`` is the single place that knows about real-time
pacing, the visualiser protocol, and the logger context.  It knows
nothing about UAV payloads, cueing rules, or world entities.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Strategy pattern for headless / interactive modes.
Implements: SIM-005 (headless), SIM-006 (real-time pacing), NF-VIZ-018,
            NF-VIZ-019.
"""
from __future__ import annotations

import time
from typing import Optional, Protocol, runtime_checkable


class SimulationEngine(Protocol):
    """
    Structural protocol matching ``SimulationEngine``'s public API.

    Typed as a Protocol so ``SimulationRunner`` can be unit-tested with
    a lightweight mock without importing the concrete engine.
    """
    config: object          # has .duration_s and .dt
    logger: object          # context-manager with __enter__ / __exit__
    sim_time: float
    step_detections: set
    track_manager: object

    def step(self) -> float: ...


@runtime_checkable
class Visualiser(Protocol):
    """
    Structural protocol for DebugPlot / SimulationView.

    Any object that exposes these three properties and a ``render()``
    method satisfies the protocol, keeping the runner decoupled from
    the concrete visualiser class.
    """

    @property
    def window_closed(self) -> bool: ...

    @property
    def paused(self) -> bool: ...

    def render(
        self,
        entities: list,
        sim_time: float = 0.0,
        detected_ids: Optional[set] = None,
        track_manager: Optional[object] = None,
    ) -> None: ...


class SimulationRunner:
    """
    Orchestrates the simulation step-loop for both headless and interactive
    modes (SIM-005, SIM-006, NF-VIZ-018, NF-VIZ-019).

    Parameters
    ----------
    engine : SimulationEngine
        The configured, populated engine to advance.
    visualiser : Visualiser, optional
        If ``None``, runs headlessly — no render calls, no GUI polling.
        If provided, renders every step and polls ``window_closed`` /
        ``paused`` on each iteration.

    Examples
    --------
    **Headless** (e.g. batch benchmarking, unit tests)::

        runner = SimulationRunner(engine)
        runner.run()

    **Interactive** (real-time visualisation)::

        plot = SimulationView(world_x, world_y, ...)
        runner = SimulationRunner(engine, visualiser=plot)
        runner.run()
    """

    def __init__(
        self,
        engine: SimulationEngine,
        visualiser: Optional[Visualiser] = None,
    ) -> None:
        self._engine: SimulationEngine = engine
        self._viz: Optional[Visualiser] = visualiser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the scenario for its configured duration.

        Step budget is consumed only when a real simulation step runs —
        pausing preserves the remaining budget rather than silently
        burning through it (NF-VIZ-019 fix).

        The logger context manager guarantees JSONL flush / close even
        if an exception propagates out of the step loop.
        """
        cfg = self._engine.config
        steps = int(cfg.duration_s / cfg.dt)

        with self._engine.logger:
            step = 0
            while step < steps:
                if self._should_stop():
                    break

                if self._is_paused():
                    self._render_paused()
                    time.sleep(cfg.dt)
                    continue                 # budget NOT consumed while paused

                wall_start = time.perf_counter()
                self._engine.step()
                self._render()

                # Real-time pacing: sleep unused dt budget (SIM-006)
                elapsed = time.perf_counter() - wall_start
                remaining = cfg.dt - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)

                step += 1  # advanced only when a real step ran

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        """
        Return True if the run loop should terminate early.

        In headless mode, never stops early (returns False).
        In interactive mode, returns True after the window is closed
        (NF-VIZ-018).
        """
        if self._viz is None:
            return False
        return self._viz.window_closed

    def _is_paused(self) -> bool:
        """
        Return True when the simulation is paused (NF-VIZ-019).

        Always False in headless mode.
        """
        if self._viz is None:
            return False
        return self._viz.paused

    def _render(self) -> None:
        """
        Render the current state if a visualiser is attached.

        No-op in headless mode.
        """
        if self._viz is None:
            return
        self._viz.render(
            self._engine.entities.living(),
            sim_time=self._engine.sim_time,
            detected_ids=self._engine.step_detections,
            track_manager=self._engine.track_manager,
        )

    def _render_paused(self) -> None:
        """
        Render without advancing the simulation (NF-VIZ-019).

        Keeps the visualiser responsive (entity inspection, zoom, pan)
        while the step counter and sim_time are frozen.
        """
        if self._viz is None:
            return
        self._viz.render(
            self._engine.entities.living(),
            sim_time=self._engine.sim_time,
            detected_ids=set(),          # no new detections while paused
            track_manager=self._engine.track_manager,
        )
