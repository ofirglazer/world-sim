"""
sim3dves.logging.logger
=======================
Structured JSONL simulation logger.

Design Pattern: Context Manager — guarantees file handles are closed
even when an exception propagates out of the simulation loop.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-004: Context Manager protocol explicitly applied.
Implements: LOG-001 (step log), LOG-002 (event log), LOG-005 (perf log).

Lazy-open fix
-------------
Previously the file was opened in ``__init__``, which meant every
``SimulationEngine(cfg, world)`` call in tests immediately opened a file
handle — even when the engine was never stepped and the logger context was
never entered.  Python's garbage collector eventually closed those handles
but emitted a ``ResourceWarning: unclosed file`` for each one.  Under
``pytest -W error`` (or PyCharm's strict warning settings) these warnings
were promoted to exceptions, causing unrelated tests to fail.

Fix: the file is now opened lazily in ``__enter__``.  ``__init__`` only
stores the path.  ``_write`` guards against the file being absent so
``log_step`` / ``log_event`` called outside a context manager are silently
dropped rather than raising ``AttributeError``.  This preserves the
existing API contract — ``with Logger(...) as log:`` still works identically
— while eliminating the resource leak in test code.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, IO, List, Optional, Type

from sim3dves.entities.base import Entity


class Logger:
    """
    Append-only JSONL structured logger for simulation events and state.

    Usage (recommended — guarantees flush/close on exception)::

        with Logger(Path("sim.jsonl")) as log:
            log.log_step(0, entities, wall_dt_s=0.004)

    FIX vs original:
    ----------------
    1. File opened in ``__init__`` with no close guard — leaked on exception.
       Replaced with context manager protocol (``__enter__`` / ``__exit__``).
    2. ``engine.run()`` called ``logger.close()`` AFTER the loop — any exception
       mid-run left the file open.  Context manager wrapping the loop fixes this.
    3. No event logging (LOG-002).
    4. No wall-clock performance column (LOG-005).
    5. No ``heading``, ``alive``, ``state``, or ``is_eoi`` fields in step record.
    6. File opened in ``__init__`` caused ResourceWarning in tests that
       constructed an engine but never entered the logger context.
       File is now opened lazily in ``__enter__`` (lazy-open fix).
    """

    def __init__(self, filepath: Path) -> None:
        """
        Store *filepath*; do NOT open the file yet.

        The file is opened in ``__enter__`` so that constructing a
        ``Logger`` (or a ``SimulationEngine`` that owns one) in a test
        does not create a file handle that must be explicitly closed.

        Parameters
        ----------
        filepath : Path
            Destination JSONL file.  Parent directory must exist at the
            time ``__enter__`` is called, not at construction time.
        """
        self._filepath: Path = filepath
        self._file: Optional[IO[str]] = None   # opened lazily in __enter__

    # ### Context Manager protocol ###

    def __enter__(self) -> "Logger":
        """
        Open the log file and return self.

        Opening here (rather than in ``__init__``) means the file handle
        is created only when the simulation loop actually starts, and is
        guaranteed to be closed by ``__exit__`` even if an exception
        propagates out of the loop.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._filepath.open("w", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Flush and close the file; do not suppress exceptions."""
        self.close()
        return False  # Propagate any exception to the caller

    # ### Public logging API ###

    def log_step(
        self,
        step: int,
        entities: List[Entity],
        wall_dt_s: float = 0.0,
    ) -> None:
        """
        Write one step record to the JSONL stream (LOG-001, LOG-005).

        Parameters
        ----------
        step : int
            Zero-based step index.
        entities : list[Entity]
            Entities to include in this record (typically ``living()``).
        wall_dt_s : float
            Wall-clock time consumed by this step in seconds (LOG-005).
        """
        record: Dict[str, Any] = {
            "step": step,
            "wall_dt_ms": round(wall_dt_s * 1_000.0, 3),  # milliseconds
            "entity_count": len(entities),
            "entities": [
                {
                    "id": e.entity_id,
                    "type": e.entity_type.name,
                    "alive": e.alive,
                    "is_eoi": e.is_eoi,
                    "pos": [round(v, 4) for v in e.position.tolist()],
                    "vel": [round(v, 6) for v in e.velocity.tolist()],
                    "heading_deg": round(e.heading, 2),
                    "state": e.state.name,
                }
                for e in entities
            ],
        }
        self._write(record)

    def log_event(self, event_dict: Dict[str, Any]) -> None:
        """
        Write a discrete event record to the JSONL stream (LOG-002).

        Parameters
        ----------
        event_dict : dict
            Arbitrary event metadata; must be JSON-serializable.
        """
        self._write({"_record_type": "event", **event_dict})

    # ### Internal helpers ###

    def _write(self, record: Dict[str, Any]) -> None:
        """
        Serialise *record* to JSON and append a newline.

        No-op if the file has not been opened yet (i.e. the logger was
        constructed but ``__enter__`` was never called).  This prevents
        ``AttributeError`` when ``log_step`` / ``log_event`` are called
        on a logger that was never used as a context manager.
        """
        if self._file is None or self._file.closed:
            return
        self._file.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle (idempotent)."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
