"""
tests.test_web
==============
Unit and integration tests for the web adapter layer.

TestSerialiser      -- pure function tests, no async needed.
TestSimulationWorker -- thread lifecycle tests, no async needed.
TestWebServer       -- FastAPI HTTP and WebSocket integration tests.

NF-TE-001: All new web modules covered.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import json
import queue
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_pedestrian(eid: str = "ped-01", is_eoi: bool = False):
    """Build a real PedestrianEntity for serialiser tests."""
    from sim3dves.entities.pedestrian import PedestrianEntity
    return PedestrianEntity(
        entity_id=eid,
        position=np.array([100.0, 200.0, 0.0]),
        velocity=np.array([1.0, 0.5, 0.0]),
        is_eoi=is_eoi,
    )


def _make_uav(eid: str = "uav-00"):
    """Build a real UAVEntity for serialiser tests."""
    from sim3dves.entities.uav import UAVEntity
    return UAVEntity(
        entity_id=eid,
        position=np.array([300.0, 300.0, 300.0]),
        world_extent=np.array([1500.0, 1500.0]),
        rng=np.random.default_rng(0),
    )


def _make_engine_mock(steps: int = 5, dt: float = 0.1) -> MagicMock:
    """Return a mock that satisfies the SimulationEngine protocol."""
    cfg = SimpleNamespace(duration_s=steps * dt, dt=dt)
    eng = MagicMock()
    eng.config = cfg
    eng.sim_time = 0.0
    eng.step_idx = 0
    eng.step_detections = set()
    eng.track_manager = MagicMock()
    eng.track_manager.active_tracks = {}
    eng.entities = MagicMock()
    eng.entities.living.return_value = []
    eng.logger.__enter__ = MagicMock(return_value=eng.logger)
    eng.logger.__exit__  = MagicMock(return_value=False)
    return eng


# ---------------------------------------------------------------------------
# 1. TestSerialiser
# ---------------------------------------------------------------------------

class TestSerialiser(unittest.TestCase):
    """Pure function tests for serialise_frame()."""

    def setUp(self) -> None:
        from sim3dves.web.serialiser import serialise_frame
        self._fn = serialise_frame

    def test_empty_frame_structure(self) -> None:
        """serialise_frame with no entities returns correct top-level keys."""
        frame = self._fn([], 0.0, 0, set())
        self.assertIn("t",        frame)
        self.assertIn("step",     frame)
        self.assertIn("entities", frame)
        self.assertIn("tracks",   frame)
        self.assertEqual(frame["entities"], [])
        self.assertEqual(frame["tracks"],   [])

    def test_sim_time_rounded(self) -> None:
        """sim_time is rounded to 2 dp."""
        frame = self._fn([], 1.23456789, 7, set())
        self.assertEqual(frame["t"], 1.23)

    def test_pedestrian_record_keys(self) -> None:
        """Pedestrian record contains base keys and no fov key."""
        ped   = _make_pedestrian()
        frame = self._fn([ped], 0.0, 0, set())
        rec   = frame["entities"][0]
        for key in ("id", "type", "pos", "vel", "heading", "is_eoi", "state", "detected"):
            self.assertIn(key, rec)
        self.assertNotIn("fov", rec)

    def test_pedestrian_type_name(self) -> None:
        """Entity type is serialised as the enum member name."""
        ped   = _make_pedestrian()
        frame = self._fn([ped], 0.0, 0, set())
        self.assertEqual(frame["entities"][0]["type"], "PEDESTRIAN")

    def test_position_rounded(self) -> None:
        """Position values are rounded to 2 dp."""
        ped   = _make_pedestrian()
        frame = self._fn([ped], 0.0, 0, set())
        pos   = frame["entities"][0]["pos"]
        for v in pos:
            self.assertEqual(v, round(v, 2))

    def test_detected_flag(self) -> None:
        """detected=True when entity_id is in detected_ids."""
        ped   = _make_pedestrian("ped-det")
        frame = self._fn([ped], 0.0, 0, {"ped-det"})
        self.assertTrue(frame["entities"][0]["detected"])

    def test_not_detected_flag(self) -> None:
        """detected=False when entity_id is NOT in detected_ids."""
        ped   = _make_pedestrian("ped-x")
        frame = self._fn([ped], 0.0, 0, set())
        self.assertFalse(frame["entities"][0]["detected"])

    def test_uav_record_has_fov(self) -> None:
        """UAV entity record contains a fov key."""
        uav   = _make_uav()
        frame = self._fn([uav], 0.0, 0, set())
        rec   = frame["entities"][0]
        self.assertIn("fov",           rec)
        self.assertIn("autopilot_mode",rec)
        self.assertIn("endurance_s",   rec)
        self.assertIn("low_fuel",      rec)

    def test_uav_type_name(self) -> None:
        """UAV entity type name is 'UAV'."""
        uav   = _make_uav()
        frame = self._fn([uav], 0.0, 0, set())
        self.assertEqual(frame["entities"][0]["type"], "UAV")

    def test_track_record_structure(self) -> None:
        """Track record contains required keys and 2x2 cov."""
        from sim3dves.payload.track_manager import TrackManager, TrackState, TrackQuality
        trk = TrackState(
            entity_id="ped-01",
            x=np.array([100.0, 200.0, 0.0, 0.0]),
            P=np.eye(4) * 25.0,
            quality=TrackQuality.HIGH,
            is_eoi=True,
        )
        tm = MagicMock(spec=TrackManager)
        tm.active_tracks = {"ped-01": trk}
        frame = self._fn([], 0.0, 0, set(), tm)
        self.assertEqual(len(frame["tracks"]), 1)
        rec = frame["tracks"][0]
        for key in ("entity_id", "quality", "pos_xy", "cov", "is_eoi", "age_steps"):
            self.assertIn(key, rec)
        # cov must be 2×2
        self.assertEqual(len(rec["cov"]),    2)
        self.assertEqual(len(rec["cov"][0]), 2)

    def test_frame_is_json_serialisable(self) -> None:
        """The returned dict must be json.dumps()-able without error."""
        uav   = _make_uav()
        ped   = _make_pedestrian()
        frame = self._fn([uav, ped], 12.3, 123, {"ped-01"})
        try:
            json.dumps(frame)
        except TypeError as exc:
            self.fail(f"json.dumps raised: {exc}")


# ---------------------------------------------------------------------------
# 2. TestSimulationWorker
# ---------------------------------------------------------------------------

class TestSimulationWorker(unittest.TestCase):
    """Thread lifecycle tests for SimulationWorker."""

    def _make_worker(self, steps: int = 5, dt: float = 0.01):
        from sim3dves.web.worker import SimulationWorker
        eng = _make_engine_mock(steps=steps, dt=dt)
        fq  = queue.Queue(maxsize=50)
        w   = SimulationWorker(eng, fq)
        return w, eng, fq

    def test_frames_enqueued(self) -> None:
        """Worker enqueues at least one frame during a 5-step run."""
        w, eng, fq = self._make_worker(steps=5, dt=0.01)
        w.start()
        w._thread.join(timeout=2.0)
        self.assertFalse(fq.empty(), "No frames were enqueued")

    def test_step_called_n_times(self) -> None:
        """engine.step() is called exactly `steps` times."""
        w, eng, fq = self._make_worker(steps=5, dt=0.01)
        w.start()
        w._thread.join(timeout=2.0)
        self.assertEqual(eng.step.call_count, 5)

    def test_stop_terminates_thread(self) -> None:
        """stop() causes the thread to exit within 1 second."""
        w, eng, fq = self._make_worker(steps=1000, dt=0.001)
        w.start()
        time.sleep(0.05)
        w.stop()
        w._thread.join(timeout=1.0)
        self.assertFalse(w.is_alive, "Thread still alive after stop()")

    def test_pause_suspends_stepping(self) -> None:
        """pause() stops new frames arriving; resume() restarts them."""
        w, eng, fq = self._make_worker(steps=200, dt=0.005)
        w.start()
        time.sleep(0.02)
        count_before = eng.step.call_count
        w.pause()
        time.sleep(0.05)
        count_paused = eng.step.call_count
        # Very few (0-2) extra steps allowed during pause() latency
        self.assertAlmostEqual(count_before, count_paused, delta=3)
        w.resume()
        time.sleep(0.05)
        self.assertGreater(eng.step.call_count, count_paused)
        w.stop()

    def test_logger_context_used(self) -> None:
        """Logger __enter__ and __exit__ are called exactly once."""
        w, eng, fq = self._make_worker(steps=3, dt=0.01)
        w.start()
        w._thread.join(timeout=2.0)
        eng.logger.__enter__.assert_called_once()
        eng.logger.__exit__.assert_called_once()

    def test_is_paused_property(self) -> None:
        """is_paused reflects the pause/resume state correctly."""
        w, eng, fq = self._make_worker(steps=100, dt=0.005)
        w.start()
        self.assertFalse(w.is_paused)
        w.pause()
        self.assertTrue(w.is_paused)
        w.resume()
        self.assertFalse(w.is_paused)
        w.stop()


# ---------------------------------------------------------------------------
# 3. TestWebServer
# ---------------------------------------------------------------------------

class TestWebServer(unittest.TestCase):
    """FastAPI HTTP and WebSocket integration tests."""

    def setUp(self) -> None:
        """Patch build_default_engine to return a fast mock engine."""
        from sim3dves.web.server import app
        from sim3dves.web.session_registry import registry
        from fastapi.testclient import TestClient

        # Replace the real engine builder with a mock that runs instantly
        self._mock_engine = _make_engine_mock(steps=50, dt=0.005)
        self._patcher = patch(
            "sim3dves.web.server.build_default_engine",
            return_value=self._mock_engine,
        )
        self._patcher.start()
        # Ensure each test starts with an empty registry
        with registry._lock:
            registry._sessions.clear()

        self._app    = app
        self._client = TestClient(app)
        self._reg    = registry

    def tearDown(self) -> None:
        self._patcher.stop()
        # Destroy any sessions left open by the test
        for session in self._reg.all_sessions():
            self._reg.destroy(session.scenario_id)

    def _create(self) -> str:
        """POST /scenarios and return the scenario_id."""
        r = self._client.post("/scenarios")
        self.assertEqual(r.status_code, 200)
        return r.json()["scenario_id"]

    def test_create_scenario_returns_id(self) -> None:
        """POST /scenarios returns a scenario_id string."""
        sid = self._create()
        self.assertIsInstance(sid, str)
        self.assertGreater(len(sid), 10)

    def test_info_endpoint_alive(self) -> None:
        """GET /scenarios/{id}/info reports alive=True immediately."""
        sid = self._create()
        r   = self._client.get(f"/scenarios/{sid}/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["alive"])

    def test_info_endpoint_world_dims(self) -> None:
        """Info endpoint returns world_x and world_y."""
        sid  = self._create()
        data = self._client.get(f"/scenarios/{sid}/info").json()
        self.assertIn("world_x", data)
        self.assertIn("world_y", data)
        self.assertGreater(data["world_x"], 0)

    def test_info_404_unknown(self) -> None:
        """GET /scenarios/bad-id/info returns 404."""
        r = self._client.get("/scenarios/does-not-exist/info")
        self.assertEqual(r.status_code, 404)

    def test_delete_scenario(self) -> None:
        """DELETE /scenarios/{id} returns 204 and removes the session."""
        sid = self._create()
        r   = self._client.delete(f"/scenarios/{sid}")
        self.assertEqual(r.status_code, 204)
        self.assertIsNone(self._reg.get(sid))

    def test_delete_unknown_404(self) -> None:
        """DELETE /scenarios/bad-id returns 404."""
        r = self._client.delete("/scenarios/not-a-real-id")
        self.assertEqual(r.status_code, 404)

    def test_websocket_hello_frame(self) -> None:
        """WebSocket connection receives a hello frame with world dims."""
        sid = self._create()
        with self._client.websocket_connect(f"/ws/sim/{sid}") as ws:
            msg  = json.loads(ws.receive_text())
            self.assertEqual(msg["type"], "hello")
            self.assertIn("world_x", msg)
            self.assertIn("world_y", msg)

    def test_websocket_unknown_scenario(self) -> None:
        """WebSocket to unknown scenario_id closes with code 4004."""
        with self.assertRaises(Exception):
            # TestClient raises on close-with-code
            with self._client.websocket_connect("/ws/sim/bad-id") as ws:
                ws.receive_text()

    def test_pause_command_pauses_worker(self) -> None:
        """Sending pause command sets worker.is_paused."""
        sid     = self._create()
        session = self._reg.get(sid)
        with self._client.websocket_connect(f"/ws/sim/{sid}") as ws:
            ws.receive_text()   # consume hello
            ws.send_text(json.dumps({"action": "pause"}))
            time.sleep(0.05)
        self.assertTrue(session.worker.is_paused)

    def test_resume_command_resumes_worker(self) -> None:
        """Sending resume after pause clears worker.is_paused."""
        sid     = self._create()
        session = self._reg.get(sid)
        session.worker.pause()
        with self._client.websocket_connect(f"/ws/sim/{sid}") as ws:
            ws.receive_text()   # consume hello
            ws.send_text(json.dumps({"action": "resume"}))
            time.sleep(0.05)
        self.assertFalse(session.worker.is_paused)


if __name__ == "__main__":
    unittest.main()
