"""
sim3dves.web.worker
===================
SimulationWorker: runs one SimulationEngine in a daemon thread,
serialises each completed step into a frame dict, and enqueues it for
the async WebSocket broadcaster.

Design Pattern: Active Object -- encapsulates a thread with its own step
loop, exposing only thread-safe control primitives (pause, resume, stop)
and a read-only output queue to the web layer.

Back-pressure policy
--------------------
queue.put_nowait() is used inside try/except queue.Full.  When the
broadcast consumer falls behind the oldest-frame-first drop is implicit
in the FIFO queue: the newest frame is silently dropped rather than
blocking the step loop.  This keeps wall-clock pacing exact regardless
of consumer speed.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

from sim3dves.web.serialiser import serialise_frame

# Maximum queued frames before drops begin (~3 s at 10 Hz default)
_QUEUE_MAXSIZE: int = 30


class SimulationWorker:
    """
    Runs one SimulationEngine in a background daemon thread.

    Parameters
    ----------
    engine : SimulationEngine
        Fully configured and populated engine.
    frame_queue : queue.Queue
        Output queue consumed by the async broadcaster in server.py.
        Pass queue.Queue(maxsize=_QUEUE_MAXSIZE) from the session registry.
    """

    def __init__(
        self,
        engine: Any,
        frame_queue: "queue.Queue[Dict[str, Any]]",
    ) -> None:
        self._engine = engine
        self._queue  = frame_queue
        # _stop set   → thread exits after current step
        # _paused set → thread sleeps one dt without stepping
        self._stop   = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public control API (all thread-safe via threading.Event)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background step thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="sim-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the step loop to exit after the current step."""
        self._stop.set()

    def pause(self) -> None:
        """Suspend stepping; frames stop arriving on the queue."""
        self._paused.set()

    def resume(self) -> None:
        """Resume stepping after a pause."""
        self._paused.clear()

    @property
    def is_alive(self) -> bool:
        """True while the worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        """True while the simulation is paused."""
        return self._paused.is_set()

    # ------------------------------------------------------------------
    # Step loop (runs on the worker thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Step loop: advance the engine, serialise, enqueue.

        The logger context manager is opened here (on the worker thread)
        so the JSONL file handle is guaranteed to flush and close when
        the thread exits, even if an exception propagates out of the loop.
        """
        cfg   = self._engine.config
        dt    = cfg.dt
        steps = int(cfg.duration_s / dt)

        with self._engine.logger:
            for _ in range(steps):
                # Stop signal checked first -- highest priority
                if self._stop.is_set():
                    break

                # Pause: sleep one dt without advancing sim time
                if self._paused.is_set():
                    time.sleep(dt)
                    continue

                wall_start = time.perf_counter()

                self._engine.step()

                frame = serialise_frame(
                    entities      = self._engine.entities.living(),
                    sim_time      = self._engine.sim_time,
                    step_idx      = self._engine.step_idx,
                    detected_ids  = self._engine.step_detections,
                    track_manager = self._engine.track_manager,
                )

                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass  # Consumer lagging -- drop frame, keep pacing

                # Real-time pacing: sleep unused dt budget (SIM-006)
                elapsed   = time.perf_counter() - wall_start
                remaining = dt - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
