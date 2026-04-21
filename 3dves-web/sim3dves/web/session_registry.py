"""
sim3dves.web.session_registry
==============================
SessionRegistry: thread-safe registry of active simulation sessions.

Each session owns exactly one SimulationWorker, one frame queue, and one
ConnectionManager.  The registry maps a UUID scenario_id to its Session.

Design Pattern: Registry -- centralised, keyed collection with a module-
level singleton so all FastAPI route handlers share the same state without
dependency-injection machinery.

This module has no FastAPI imports.  ConnectionManager is imported lazily
via a TYPE_CHECKING guard to avoid a circular import with server.py.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sim3dves.web.worker import SimulationWorker, _QUEUE_MAXSIZE

if TYPE_CHECKING:
    # Imported only for type checking -- avoids circular import at runtime
    from sim3dves.web.server import ConnectionManager


@dataclass
class Session:
    """
    All state for one active simulation scenario.

    Attributes
    ----------
    scenario_id : str
        UUID string key used in WebSocket URLs and REST endpoints.
    worker : SimulationWorker
        Background thread running the engine.
    frame_queue : queue.Queue
        FIFO buffer between worker and async broadcaster.
    connection_manager : ConnectionManager
        Fan-out registry of WebSocket clients watching this scenario.
    world_x : float
        World X extent in metres -- sent to the browser for view init.
    world_y : float
        World Y extent in metres.
    """
    scenario_id:        str
    worker:             SimulationWorker
    frame_queue:        "queue.Queue[Dict[str, Any]]"
    connection_manager: Any           # ConnectionManager (Any to avoid circular import)
    world_x:            float = 1500.0
    world_y:            float = 1500.0


class SessionRegistry:
    """
    Thread-safe registry of active simulation sessions.

    All public methods acquire ``_lock`` before reading or writing
    ``_sessions`` so they are safe to call from both the async event loop
    (FastAPI route handlers) and background threads.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        engine: Any,
        connection_manager: Any,
        world_x: float = 1500.0,
        world_y: float = 1500.0,
    ) -> str:
        """
        Register a new session, start its worker, and return the scenario_id.

        Parameters
        ----------
        engine : SimulationEngine
            Fully configured and populated engine.
        connection_manager : ConnectionManager
            Fresh ConnectionManager instance for this scenario.
        world_x, world_y : float
            World extents in metres, forwarded to the browser.

        Returns
        -------
        str
            UUID scenario_id used in all subsequent API calls.
        """
        scenario_id = str(uuid.uuid4())
        fq: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        worker = SimulationWorker(engine, fq)

        session = Session(
            scenario_id        = scenario_id,
            worker             = worker,
            frame_queue        = fq,
            connection_manager = connection_manager,
            world_x            = world_x,
            world_y            = world_y,
        )

        with self._lock:
            self._sessions[scenario_id] = session

        worker.start()
        return scenario_id

    def get(self, scenario_id: str) -> Optional[Session]:
        """Return the Session for *scenario_id*, or None if not found."""
        with self._lock:
            return self._sessions.get(scenario_id)

    def destroy(self, scenario_id: str) -> bool:
        """
        Stop the worker and remove the session.

        Parameters
        ----------
        scenario_id : str
            Session to remove.

        Returns
        -------
        bool
            True if the session existed and was destroyed, False if not found.
        """
        with self._lock:
            session = self._sessions.pop(scenario_id, None)

        if session is None:
            return False

        session.worker.stop()
        # Give the thread up to 5 s to flush the logger and exit cleanly
        if session.worker._thread is not None:
            session.worker._thread.join(timeout=5.0)

        return True

    def all_sessions(self) -> List[Session]:
        """Return a snapshot list of all active sessions (thread-safe)."""
        with self._lock:
            return list(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


# Module-level singleton -- imported by server.py and tests
registry = SessionRegistry()
