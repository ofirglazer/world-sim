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
"""
from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

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
    """

    def __init__(self, filepath: Path) -> None:
        """
        Open *filepath* for writing.

        Parameters
        ----------
        filepath : Path
            Destination JSONL file.  Parent directory must exist.
        """
        self._filepath: Path = filepath
        # Open immediately so callers get an error before the sim starts,
        # not midway through a 30-minute run.
        self._file = filepath.open("w", encoding="utf-8")

    # ### Context Manager protocol ###

    def __enter__(self) -> "Logger":
        """Return self so callers can use ``with Logger(...) as log:``."""
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
        """Serialize *record* to JSON and append a newline."""
        self._file.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle (idempotent)."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
