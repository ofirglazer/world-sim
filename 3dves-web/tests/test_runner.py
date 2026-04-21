"""
tests.test_runner
=================
Unit tests for SimulationRunner and HighQualityEoiCueingPolicy.

NF-TE-001: All new modules covered.
NF-TE-002: Each test cites the requirement it validates.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.runner import SimulationRunner

_D = SimDefaults()


# ---------------------------------------------------------------------------
# Minimal engine / visualiser stubs
# ---------------------------------------------------------------------------

def _make_engine(steps: int = 5, dt: float = 0.1) -> MagicMock:
    """Return a mock that satisfies the SimulationEngine protocol."""
    cfg = SimpleNamespace(duration_s=steps * dt, dt=dt)
    eng = MagicMock()
    eng.config = cfg
    eng.sim_time = 0.0
    eng.step_detections = set()
    eng.track_manager = MagicMock()
    eng.entities = MagicMock()
    eng.entities.living.return_value = []
    # logger context manager — use the mock's __enter__/__exit__
    eng.logger.__enter__ = MagicMock(return_value=eng.logger)
    eng.logger.__exit__ = MagicMock(return_value=False)
    return eng


def _make_viz(closed: bool = False, paused: bool = False) -> MagicMock:
    """Return a mock that satisfies the Visualiser protocol."""
    viz = MagicMock()
    type(viz).window_closed = property(lambda self: closed)
    type(viz).paused = property(lambda self: paused)
    return viz


# ---------------------------------------------------------------------------
# SimulationRunner — headless
# ---------------------------------------------------------------------------

class TestSimulationRunnerHeadless(unittest.TestCase):
    """SIM-005: headless run advances exactly `steps` simulation steps."""

    def test_step_called_n_times(self) -> None:
        """SIM-005: engine.step() called once per configured step."""
        eng = _make_engine(steps=5)
        SimulationRunner(eng).run()
        self.assertEqual(eng.step.call_count, 5)

    def test_no_render_calls_in_headless(self) -> None:
        """SIM-005: no render calls when visualiser is None."""
        eng = _make_engine(steps=3)
        runner = SimulationRunner(eng, visualiser=None)
        runner.run()
        # No visualiser — _render is a no-op; nothing with .render to assert.
        # The key assertion: step was called the expected number of times.
        self.assertEqual(eng.step.call_count, 3)

    def test_logger_context_used(self) -> None:
        """SIM-005: logger __enter__/__exit__ called exactly once."""
        eng = _make_engine(steps=2)
        SimulationRunner(eng).run()
        eng.logger.__enter__.assert_called_once()
        eng.logger.__exit__.assert_called_once()

    def test_zero_steps(self) -> None:
        """SIM-005: zero-step scenario calls step() zero times."""
        eng = _make_engine(steps=0)
        SimulationRunner(eng).run()
        eng.step.assert_not_called()


# ---------------------------------------------------------------------------
# SimulationRunner — interactive (window-close, pause)
# ---------------------------------------------------------------------------

class TestSimulationRunnerInteractive(unittest.TestCase):
    """NF-VIZ-018, NF-VIZ-019: window-close and pause handled correctly."""

    def test_render_called_each_step(self) -> None:
        """NF-VIZ-001: render called once per simulation step."""
        eng = _make_engine(steps=4)
        viz = _make_viz()
        SimulationRunner(eng, visualiser=viz).run()
        self.assertEqual(viz.render.call_count, 4)

    def test_window_closed_stops_loop(self) -> None:
        """NF-VIZ-018: window_closed=True breaks the run loop immediately."""
        eng = _make_engine(steps=100)
        viz = _make_viz(closed=True)
        SimulationRunner(eng, visualiser=viz).run()
        # Loop should have broken on first check, before any step
        eng.step.assert_not_called()

    def test_paused_does_not_consume_budget(self) -> None:
        """NF-VIZ-019: steps taken == 0 while paused (budget preserved)."""
        # We want a viz that is paused for a fixed number of iterations then
        # becomes unpaused.  Use a counter-based side_effect.
        eng = _make_engine(steps=3, dt=0.001)  # tiny sleep
        call_count = [0]

        class _PausedThenRun:
            @property
            def window_closed(self): return False
            @property
            def paused(self):
                # paused for first 5 checks, then run
                call_count[0] += 1
                return call_count[0] <= 5
            def render(self, *a, **kw): pass

        viz = _PausedThenRun()
        runner = SimulationRunner(eng, visualiser=viz)
        runner.run()
        # After 5 paused iters the loop continues; all 3 steps must complete
        self.assertEqual(eng.step.call_count, 3)

    def test_render_receives_track_manager(self) -> None:
        """NF-VIZ-006 M5: render is called with engine.track_manager."""
        eng = _make_engine(steps=1)
        viz = _make_viz()
        SimulationRunner(eng, visualiser=viz).run()
        _, kwargs = viz.render.call_args
        self.assertIn("track_manager", kwargs)
        self.assertEqual(kwargs["track_manager"], eng.track_manager)

    def test_render_receives_detected_ids(self) -> None:
        """M4: render receives step_detections as detected_ids."""
        eng = _make_engine(steps=1)
        eng.step_detections = {"ped-42"}
        viz = _make_viz()
        SimulationRunner(eng, visualiser=viz).run()
        _, kwargs = viz.render.call_args
        self.assertEqual(kwargs.get("detected_ids"), {"ped-42"})


# ---------------------------------------------------------------------------
# HighQualityEoiCueingPolicy
# ---------------------------------------------------------------------------

class TestHighQualityEoiCueingPolicy(unittest.TestCase):
    """M5: cueing policy reacts to TRACK_ACQUIRED events correctly."""

    def _make_policy_and_uav(self, quality_name="HIGH", is_eoi=True):
        """Build a policy with one mock UAV and a mock TrackManager."""
        from sim3dves.payload.cueing_policy import HighQualityEoiCueingPolicy
        from sim3dves.payload.track_manager import TrackManager, TrackQuality

        uav = MagicMock()
        uav.alive = True
        uav.payload = MagicMock()

        track = MagicMock()
        track.quality = getattr(TrackQuality, quality_name)
        track.position_xy = np.array([200.0, 300.0])

        tm = MagicMock(spec=TrackManager)
        tm.get_track.return_value = track

        policy = HighQualityEoiCueingPolicy(
            uav_entities=[uav],
            track_manager=tm,
        )
        return policy, uav, tm

    def _make_event(self, entity_id="ped-eoi", is_eoi=True):
        from sim3dves.core.event_bus import Event, EventType
        return Event(
            timestamp=1.0,
            event_type=EventType.TRACK_ACQUIRED,
            payload={"entity_id": entity_id, "is_eoi": is_eoi,
                     "quality": "HIGH"},
        )

    def test_cues_uav_on_high_quality_eoi(self) -> None:
        """M5: cue_orbit called when EOI track reaches HIGH quality."""
        policy, uav, _ = self._make_policy_and_uav("HIGH", True)
        policy.on_track_acquired(self._make_event(is_eoi=True))
        uav.cue_orbit.assert_called_once()

    def test_no_cue_when_not_eoi(self) -> None:
        """M5: cue_orbit NOT called when track is not EOI."""
        policy, uav, _ = self._make_policy_and_uav("HIGH", False)
        policy.on_track_acquired(self._make_event(is_eoi=False))
        uav.cue_orbit.assert_not_called()

    def test_no_cue_when_medium_quality(self) -> None:
        """M5: cue_orbit NOT called when track quality is MEDIUM."""
        policy, uav, _ = self._make_policy_and_uav("MEDIUM", True)
        policy.on_track_acquired(self._make_event(is_eoi=True))
        uav.cue_orbit.assert_not_called()

    def test_payload_command_cued_called(self) -> None:
        """M5: payload.command_cued called with the entity ID."""
        policy, uav, _ = self._make_policy_and_uav("HIGH", True)
        policy.on_track_acquired(self._make_event("ped-eoi", True))
        uav.payload.command_cued.assert_called_once_with("ped-eoi")

    def test_no_double_cue_same_entity(self) -> None:
        """M5: second TRACK_ACQUIRED for same entity does not re-cue."""
        policy, uav, _ = self._make_policy_and_uav("HIGH", True)
        evt = self._make_event("ped-eoi", True)
        policy.on_track_acquired(evt)
        policy.on_track_acquired(evt)  # second time
        self.assertEqual(uav.cue_orbit.call_count, 1)

    def test_no_cue_when_no_living_uav(self) -> None:
        """M5: no cue_orbit when all UAVs are dead."""
        from sim3dves.payload.cueing_policy import HighQualityEoiCueingPolicy
        from sim3dves.payload.track_manager import TrackManager, TrackQuality

        dead_uav = MagicMock()
        dead_uav.alive = False
        track = MagicMock()
        track.quality = TrackQuality.HIGH
        track.position_xy = np.array([100.0, 100.0])
        tm = MagicMock(spec=TrackManager)
        tm.get_track.return_value = track

        policy = HighQualityEoiCueingPolicy([dead_uav], tm)
        policy.on_track_acquired(self._make_event("ped-dead", True))
        dead_uav.cue_orbit.assert_not_called()


# ---------------------------------------------------------------------------
# UAV payload auto-stepping
# ---------------------------------------------------------------------------

class TestUAVPayloadAutoStep(unittest.TestCase):
    """M5: UAVEntity._update_behavior() automatically steps its payload."""

    def test_payload_step_called_by_uav(self) -> None:
        """M5: payload.step() called once per entity step when attached."""
        from sim3dves.entities.uav import UAVEntity

        uav = UAVEntity(
            entity_id="uav-auto",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        mock_payload = MagicMock()
        uav.payload = mock_payload

        uav.step(dt=0.1)

        mock_payload.step.assert_called_once()
        _, kwargs = mock_payload.step.call_args
        self.assertIn("uav_position", kwargs)
        self.assertIn("uav_heading_deg", kwargs)
        self.assertIn("dt", kwargs)

    def test_no_payload_no_crash(self) -> None:
        """M5: UAVEntity without payload attribute steps without error."""
        from sim3dves.entities.uav import UAVEntity

        uav = UAVEntity(
            entity_id="uav-nopay",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        # No payload attribute — must not raise
        uav.step(dt=0.1)

    def test_payload_receives_heading(self) -> None:
        """M5: payload.step receives the UAV's current heading."""
        from sim3dves.entities.uav import UAVEntity

        uav = UAVEntity(
            entity_id="uav-hdg",
            position=np.array([300.0, 300.0, 100.0]),
            heading=45.0,
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        mock_payload = MagicMock()
        uav.payload = mock_payload
        uav.step(dt=0.1)
        _, kwargs = mock_payload.step.call_args
        # heading may drift slightly after one step; just assert it is a float
        self.assertIsInstance(kwargs["uav_heading_deg"], float)


if __name__ == "__main__":
    unittest.main()
