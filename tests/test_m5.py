"""
tests.test_m5
=============
Unit tests for M5: TrackManager Kalman filter, track lifecycle, quality
grading, TRACK_ACQUIRED / TRACK_LOST events, engine integration, and
visualiser covariance ellipse geometry.

NF-TE-001: All new modules covered; ≥ 80% coverage.
NF-TE-002: Every test docstring cites the Req ID(s) validated.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import math
import unittest
from typing import Dict, List
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")
import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.uav import UAVEntity
from sim3dves.payload.track_manager import TrackManager, TrackQuality, TrackState

_D = SimDefaults()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(
    entity_id: str = "ped-0",
    pos: np.ndarray | None = None,
    is_eoi: bool = False,
) -> MagicMock:
    """Minimal mock entity for TrackManager tests."""
    e = MagicMock()
    e.entity_id = entity_id
    e.position = pos if pos is not None else np.array([100.0, 200.0, 0.0])
    e.velocity = np.zeros(3)
    e.is_eoi = is_eoi
    e.alive = True
    return e


def _make_tm() -> TrackManager:
    """TrackManager with no callbacks (for isolation)."""
    return TrackManager()


# ---------------------------------------------------------------------------
# 1. Kalman filter — predict
# ---------------------------------------------------------------------------

class TestKalmanPredict(unittest.TestCase):
    """M5: Kalman predict step propagates state correctly."""

    def setUp(self) -> None:
        self.tm = _make_tm()

    def test_F_shape(self) -> None:
        """M5: transition matrix is 4×4."""
        F = self.tm._build_F(dt=0.1)
        self.assertEqual(F.shape, (4, 4))

    def test_F_constant_velocity(self) -> None:
        """M5: F propagates position by v*dt with constant velocity model."""
        dt = 0.1
        F = self.tm._build_F(dt)
        x = np.array([100.0, 200.0, 10.0, 5.0])  # pos=(100,200), vel=(10,5)
        x_pred = F @ x
        self.assertAlmostEqual(x_pred[0], 100.0 + 10.0 * dt, places=6)
        self.assertAlmostEqual(x_pred[1], 200.0 + 5.0 * dt,  places=6)
        self.assertAlmostEqual(x_pred[2], 10.0, places=6)   # vx unchanged
        self.assertAlmostEqual(x_pred[3], 5.0,  places=6)   # vy unchanged

    def test_Q_shape(self) -> None:
        """M5: process noise matrix is 4×4 symmetric positive semi-definite."""
        Q = self.tm._build_Q(dt=0.1)
        self.assertEqual(Q.shape, (4, 4))
        # Symmetric
        np.testing.assert_array_almost_equal(Q, Q.T)
        # PSD: all eigenvalues ≥ 0
        eigvals = np.linalg.eigvalsh(Q)
        self.assertTrue(np.all(eigvals >= -1e-10))

    def test_Q_scales_with_dt(self) -> None:
        """M5: larger dt produces larger process noise."""
        Q_small = self.tm._build_Q(dt=0.1)
        Q_large = self.tm._build_Q(dt=1.0)
        self.assertGreater(np.trace(Q_large), np.trace(Q_small))

    def test_predict_grows_covariance(self) -> None:
        """M5: predict step increases position covariance via Q contribution."""
        track = TrackState(
            entity_id="t0",
            x=np.zeros(4),
            P=np.eye(4),
        )
        P_before = track.P.copy()
        # Simulate one predict via the public step() with no detections
        self.tm.active_tracks["t0"] = track
        self.tm.step(detected_ids=set(), entities=[], dt=0.1)
        P_after = self.tm.active_tracks["t0"].P
        self.assertGreater(np.trace(P_after), np.trace(P_before))


# ---------------------------------------------------------------------------
# 2. Kalman filter — update
# ---------------------------------------------------------------------------

class TestKalmanUpdate(unittest.TestCase):
    """M5: Kalman measurement update reduces covariance."""

    def setUp(self) -> None:
        self.tm = _make_tm()

    def test_update_reduces_covariance_trace(self) -> None:
        """M5: measurement update reduces position uncertainty."""
        track = TrackState(
            entity_id="t0",
            x=np.array([100.0, 200.0, 0.0, 0.0]),
            P=np.eye(4) * 100.0,
        )
        P_before_trace = np.trace(track.P)
        z = np.array([102.0, 198.0])   # nearby measurement
        self.tm._kf_update(track, z)
        self.assertLess(np.trace(track.P), P_before_trace)

    def test_update_moves_state_toward_measurement(self) -> None:
        """M5: state estimate moves toward the measurement after update."""
        track = TrackState(
            entity_id="t0",
            x=np.array([100.0, 200.0, 0.0, 0.0]),
            P=np.eye(4) * 100.0,
        )
        z = np.array([120.0, 220.0])   # measurement is 20m away
        self.tm._kf_update(track, z)
        # Updated x should be closer to z than before
        dist_after  = float(np.linalg.norm(track.x[:2] - z))
        dist_before = float(np.linalg.norm(np.array([100.0, 200.0]) - z))
        self.assertLess(dist_after, dist_before)

    def test_R_shape(self) -> None:
        """M5: measurement noise matrix is 2×2 diagonal."""
        R = self.tm._build_R()
        self.assertEqual(R.shape, (2, 2))
        self.assertAlmostEqual(R[0, 1], 0.0)
        self.assertAlmostEqual(R[1, 0], 0.0)

    def test_P_remains_symmetric_after_update(self) -> None:
        """M5: Joseph-form update preserves covariance symmetry."""
        track = TrackState(
            entity_id="t0",
            x=np.array([100.0, 200.0, 5.0, 3.0]),
            P=np.eye(4) * 50.0,
        )
        self.tm._kf_update(track, np.array([105.0, 198.0]))
        np.testing.assert_array_almost_equal(track.P, track.P.T, decimal=10)

    def test_P_remains_positive_definite_after_update(self) -> None:
        """M5: covariance eigenvalues stay positive after update."""
        track = TrackState(
            entity_id="t0",
            x=np.array([100.0, 200.0, 5.0, 3.0]),
            P=np.eye(4) * 50.0,
        )
        self.tm._kf_update(track, np.array([105.0, 198.0]))
        eigvals = np.linalg.eigvalsh(track.P)
        self.assertTrue(np.all(eigvals > 0))


# ---------------------------------------------------------------------------
# 3. Track quality grading
# ---------------------------------------------------------------------------

class TestTrackQuality(unittest.TestCase):
    """M5: quality tier determined by consecutive hit count."""

    def setUp(self) -> None:
        self.tm = _make_tm()

    def _track_with_hits(self, n: int) -> TrackState:
        t = TrackState("x", np.zeros(4), np.eye(4))
        t.hit_count = n
        return t

    def test_zero_hits_is_low(self) -> None:
        """M5: zero consecutive hits → LOW quality."""
        self.assertEqual(
            self.tm._grade(self._track_with_hits(0)), TrackQuality.LOW
        )

    def test_one_hit_is_low(self) -> None:
        """M5: 1 consecutive hit below MEDIUM threshold → LOW."""
        if _D.TRK_MEDIUM_HIT_COUNT > 1:
            self.assertEqual(
                self.tm._grade(self._track_with_hits(1)), TrackQuality.LOW
            )

    def test_medium_threshold(self) -> None:
        """M5: TRK_MEDIUM_HIT_COUNT hits → MEDIUM quality."""
        self.assertEqual(
            self.tm._grade(self._track_with_hits(_D.TRK_MEDIUM_HIT_COUNT)),
            TrackQuality.MEDIUM,
        )

    def test_high_threshold(self) -> None:
        """M5: TRK_HIGH_HIT_COUNT hits → HIGH quality."""
        self.assertEqual(
            self.tm._grade(self._track_with_hits(_D.TRK_HIGH_HIT_COUNT)),
            TrackQuality.HIGH,
        )

    def test_above_high_stays_high(self) -> None:
        """M5: more than TRK_HIGH_HIT_COUNT hits → stays HIGH."""
        self.assertEqual(
            self.tm._grade(self._track_with_hits(_D.TRK_HIGH_HIT_COUNT + 10)),
            TrackQuality.HIGH,
        )


# ---------------------------------------------------------------------------
# 4. Track lifecycle — new tracks, misses, removal
# ---------------------------------------------------------------------------

class TestTrackLifecycle(unittest.TestCase):
    """M5: track is created on detection, degrades on miss, removed when lost."""

    def setUp(self) -> None:
        self.tm = _make_tm()

    def test_new_track_created_on_first_detection(self) -> None:
        """M5: entity detected for the first time creates a new TrackState."""
        ent = _make_entity("ped-1")
        self.tm.step({"ped-1"}, [ent], dt=0.1)
        self.assertIn("ped-1", self.tm.active_tracks)

    def test_track_state_initial_position(self) -> None:
        """M5: initial track position matches entity measurement."""
        pos = np.array([150.0, 250.0, 0.0])
        ent = _make_entity("ped-2", pos=pos)
        self.tm.step({"ped-2"}, [ent], dt=0.1)
        ts = self.tm.active_tracks["ped-2"]
        self.assertAlmostEqual(ts.x[0], 150.0, places=1)
        self.assertAlmostEqual(ts.x[1], 250.0, places=1)

    def test_miss_increments_miss_count(self) -> None:
        """M5: step with no detection increments miss_count."""
        ent = _make_entity("ped-3")
        self.tm.step({"ped-3"}, [ent], dt=0.1)  # create track
        self.tm.step(set(), [], dt=0.1)           # miss
        ts = self.tm.active_tracks["ped-3"]
        self.assertEqual(ts.miss_count, 1)

    def test_hit_resets_miss_count(self) -> None:
        """M5: detection after miss resets miss_count to zero."""
        ent = _make_entity("ped-4")
        self.tm.step({"ped-4"}, [ent], dt=0.1)
        self.tm.step(set(), [], dt=0.1)           # miss
        self.tm.step({"ped-4"}, [ent], dt=0.1)   # hit
        self.assertEqual(self.tm.active_tracks["ped-4"].miss_count, 0)

    def test_track_removed_after_max_misses(self) -> None:
        """M5: track is removed after TRK_MAX_MISS_STEPS consecutive misses."""
        ent = _make_entity("ped-5")
        self.tm.step({"ped-5"}, [ent], dt=0.1)
        for _ in range(_D.TRK_MAX_MISS_STEPS):
            self.tm.step(set(), [], dt=0.1)
        self.assertNotIn("ped-5", self.tm.active_tracks)

    def test_track_survives_before_max_misses(self) -> None:
        """M5: track survives until miss_count reaches TRK_MAX_MISS_STEPS."""
        ent = _make_entity("ped-6")
        self.tm.step({"ped-6"}, [ent], dt=0.1)
        for _ in range(_D.TRK_MAX_MISS_STEPS - 1):
            self.tm.step(set(), [], dt=0.1)
        self.assertIn("ped-6", self.tm.active_tracks)

    def test_age_advances_each_step(self) -> None:
        """M5: age_steps increments by 1 each step for an active track."""
        ent = _make_entity("ped-7")
        self.tm.step({"ped-7"}, [ent], dt=0.1)
        for _ in range(4):
            self.tm.step({"ped-7"}, [ent], dt=0.1)
        # age_steps is not advanced on the creation step; 4 subsequent steps -> age=4
        self.assertEqual(self.tm.active_tracks["ped-7"].age_steps, 4)

    def test_multiple_tracks_independent(self) -> None:
        """M5: two tracked entities do not interfere with each other."""
        e1 = _make_entity("a", pos=np.array([10.0, 10.0, 0.0]))
        e2 = _make_entity("b", pos=np.array([500.0, 500.0, 0.0]))
        self.tm.step({"a", "b"}, [e1, e2], dt=0.1)
        self.tm.step({"a"}, [e1, e2], dt=0.1)    # b misses
        self.assertEqual(self.tm.active_tracks["b"].miss_count, 1)
        self.assertEqual(self.tm.active_tracks["a"].miss_count, 0)


# ---------------------------------------------------------------------------
# 5. TRACK_ACQUIRED / TRACK_LOST callbacks
# ---------------------------------------------------------------------------

class TestTrackCallbacks(unittest.TestCase):
    """M5: callbacks fire at correct lifecycle transitions."""

    def test_acquired_fires_at_medium_quality(self) -> None:
        """M5: TRACK_ACQUIRED callback fires when quality first reaches MEDIUM."""
        acquired: list = []
        tm = TrackManager(on_track_acquired=lambda eid, ts: acquired.append(eid))
        ent = _make_entity("ped-cb")
        # Step enough times to reach MEDIUM quality
        for _ in range(_D.TRK_MEDIUM_HIT_COUNT + 1):
            tm.step({"ped-cb"}, [ent], dt=0.1)
        self.assertIn("ped-cb", acquired)

    def test_acquired_fires_once(self) -> None:
        """M5: TRACK_ACQUIRED fires exactly once per LOW->MEDIUM transition."""
        acquired: list = []
        tm = TrackManager(on_track_acquired=lambda eid, ts: acquired.append(eid))
        ent = _make_entity("ped-once")
        for _ in range(_D.TRK_HIGH_HIT_COUNT + 5):
            tm.step({"ped-once"}, [ent], dt=0.1)
        # Should fire exactly once (at MEDIUM, not again at HIGH)
        self.assertEqual(acquired.count("ped-once"), 1)

    def test_lost_fires_after_max_misses(self) -> None:
        """M5: TRACK_LOST callback fires when track is removed."""
        lost: list = []
        tm = TrackManager(on_track_lost=lambda eid, ts: lost.append(eid))
        ent = _make_entity("ped-lost")
        tm.step({"ped-lost"}, [ent], dt=0.1)
        for _ in range(_D.TRK_MAX_MISS_STEPS):
            tm.step(set(), [], dt=0.1)
        self.assertIn("ped-lost", lost)

    def test_track_state_passed_to_callback(self) -> None:
        """M5: callback receives a valid TrackState object."""
        received: list = []
        tm = TrackManager(on_track_acquired=lambda eid, ts: received.append(ts))
        ent = _make_entity("ped-state")
        for _ in range(_D.TRK_MEDIUM_HIT_COUNT + 1):
            tm.step({"ped-state"}, [ent], dt=0.1)
        self.assertTrue(len(received) >= 1)
        self.assertIsInstance(received[0], TrackState)

    def test_no_callback_no_error(self) -> None:
        """M5: TrackManager works without callback (no exception raised)."""
        tm = TrackManager()  # no callbacks
        ent = _make_entity("ped-nocb")
        for _ in range(_D.TRK_MEDIUM_HIT_COUNT + 1):
            tm.step({"ped-nocb"}, [ent], dt=0.1)
        for _ in range(_D.TRK_MAX_MISS_STEPS):
            tm.step(set(), [], dt=0.1)
        # No exception means pass


# ---------------------------------------------------------------------------
# 6. EOI track filtering
# ---------------------------------------------------------------------------

class TestEoiTracks(unittest.TestCase):
    """M5: eoi_tracks() returns only EOI-flagged active tracks."""

    def test_eoi_tracks_empty_when_none(self) -> None:
        """M5: eoi_tracks() empty when no tracks are EOI."""
        tm = _make_tm()
        ent = _make_entity("ped-0", is_eoi=False)
        tm.step({"ped-0"}, [ent], dt=0.1)
        self.assertEqual(len(tm.eoi_tracks()), 0)

    def test_eoi_tracks_returns_eoi_only(self) -> None:
        """M5: eoi_tracks() returns only tracks with is_eoi=True."""
        tm = _make_tm()
        e_eoi  = _make_entity("eoi-1", is_eoi=True)
        e_norm = _make_entity("norm-1", is_eoi=False)
        tm.step({"eoi-1", "norm-1"}, [e_eoi, e_norm], dt=0.1)
        eoi_ids = [t.entity_id for t in tm.eoi_tracks()]
        self.assertIn("eoi-1",  eoi_ids)
        self.assertNotIn("norm-1", eoi_ids)


# ---------------------------------------------------------------------------
# 7. TrackState helpers
# ---------------------------------------------------------------------------

class TestTrackStateHelpers(unittest.TestCase):
    """M5: TrackState property helpers return correct sub-views."""

    def _make_track(self) -> TrackState:
        x = np.array([10.0, 20.0, 3.0, 4.0])
        P = np.diag([9.0, 16.0, 1.0, 1.0])
        return TrackState("t", x, P)

    def test_position_xy(self) -> None:
        """M5: position_xy returns the first two state elements."""
        ts = self._make_track()
        np.testing.assert_array_almost_equal(ts.position_xy, [10.0, 20.0])

    def test_velocity_xy(self) -> None:
        """M5: velocity_xy returns the last two state elements."""
        ts = self._make_track()
        np.testing.assert_array_almost_equal(ts.velocity_xy, [3.0, 4.0])

    def test_position_covariance_shape(self) -> None:
        """M5: position_covariance is a 2×2 sub-block of P."""
        ts = self._make_track()
        self.assertEqual(ts.position_covariance.shape, (2, 2))

    def test_position_covariance_values(self) -> None:
        """M5: position_covariance matches P[:2,:2]."""
        ts = self._make_track()
        expected = np.diag([9.0, 16.0])
        np.testing.assert_array_almost_equal(ts.position_covariance, expected)


# ---------------------------------------------------------------------------
# 8. Engine integration
# ---------------------------------------------------------------------------

class TestEngineTrackManager(unittest.TestCase):
    """M5: engine creates TrackManager and advances it each step."""

    def _make_world(self):
        from sim3dves.core.world import World
        return World(extent=np.array([600.0, 600.0]))

    def test_engine_has_track_manager(self) -> None:
        """M5: SimulationEngine exposes a TrackManager instance."""
        from sim3dves.core.engine import SimulationConfig, SimulationEngine
        sim = SimulationEngine(SimulationConfig(logging_enabled=False), self._make_world())
        self.assertIsInstance(sim.track_manager, TrackManager)

    def test_track_created_via_engine_step(self) -> None:
        """M5: detection pre-loaded into payload buffer reaches TrackManager via engine step.

        Payload stepping is now automatic (UAVEntity._update_behavior step 7),
        but we pre-load _pending_detections directly to test the engine's
        flush-and-track path in isolation.
        """
        from sim3dves.core.engine import SimulationConfig, SimulationEngine
        from sim3dves.payload.optical_payload import OpticalPayload

        world = self._make_world()
        sim = SimulationEngine(
            SimulationConfig(logging_enabled=False), world
        )
        uav = UAVEntity(
            entity_id="uav-trk",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        # Pre-load a detection so it fires on the next step
        payload = OpticalPayload(owner_id="uav-trk")
        payload._pending_detections = [{
            "uav_id": "uav-trk", "target_id": "ped-trk",
            "pd": 1.0, "los_clear": True,
        }]
        uav.payload = payload
        sim.add_entity(uav)

        # Also add a mock pedestrian so the entity_map lookup works
        from sim3dves.entities.pedestrian import PedestrianEntity
        ped = PedestrianEntity(
            entity_id="ped-trk",
            position=np.array([310.0, 305.0, 0.0]),
            velocity=np.zeros(3),
        )
        sim.add_entity(ped)

        with sim.logger:
            sim.step()

        self.assertIn("ped-trk", sim.track_manager.active_tracks)

    def test_track_acquired_event_published(self) -> None:
        """M5: TRACK_ACQUIRED event reaches EventBus when track reaches MEDIUM."""
        from sim3dves.core.engine import SimulationConfig, SimulationEngine
        from sim3dves.core.event_bus import EventType
        from sim3dves.payload.optical_payload import OpticalPayload
        from sim3dves.entities.pedestrian import PedestrianEntity

        world = self._make_world()
        sim = SimulationEngine(
            SimulationConfig(logging_enabled=False), world
        )
        acquired: list = []
        sim.event_bus.subscribe(
            EventType.TRACK_ACQUIRED, lambda e: acquired.append(e)
        )

        ped = PedestrianEntity(
            entity_id="ped-acq",
            position=np.array([310.0, 305.0, 0.0]),
            velocity=np.zeros(3),
        )
        sim.add_entity(ped)

        uav = UAVEntity(
            entity_id="uav-acq",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        uav.payload = OpticalPayload(owner_id="uav-acq")
        sim.add_entity(uav)

        # Drive enough steps to reach MEDIUM quality (TRK_MEDIUM_HIT_COUNT + 1)
        with sim.logger:
            for _ in range(_D.TRK_MEDIUM_HIT_COUNT + 2):
                uav.payload._pending_detections = [{
                    "uav_id": "uav-acq", "target_id": "ped-acq",
                    "pd": 1.0, "los_clear": True,
                }]
                sim.step()

        self.assertGreaterEqual(len(acquired), 1)


# ---------------------------------------------------------------------------
# 9. Visualiser ellipse geometry
# ---------------------------------------------------------------------------

class TestVisualizerEllipse(unittest.TestCase):
    """NF-VIZ-006 M5: covariance ellipse geometry is correct."""

    def setUp(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sim3dves.viz.debug_plot import DebugPlot
        self.plot = DebugPlot(600.0, 600.0)

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_draw_track_ellipses_no_crash(self) -> None:
        """NF-VIZ-006 M5: _draw_track_ellipses runs without exception."""
        tm = TrackManager()
        ent = _make_entity("ped-viz")
        tm.step({"ped-viz"}, [ent], dt=0.1)
        # Should not raise
        self.plot._draw_track_ellipses(tm)

    def test_no_ellipses_on_empty_manager(self) -> None:
        """NF-VIZ-006 M5: empty TrackManager draws nothing (no patches)."""
        import matplotlib.pyplot as plt
        patches_before = len(self.plot._ax.patches)
        self.plot._draw_track_ellipses(TrackManager())
        self.assertEqual(len(self.plot._ax.patches), patches_before)

    def test_ellipse_drawn_per_track(self) -> None:
        """NF-VIZ-006 M5: one Ellipse patch is added per active track."""
        import matplotlib.patches as mp
        tm = TrackManager()
        for i in range(3):
            ent = _make_entity(f"t{i}", pos=np.array([i * 100.0, 100.0, 0.0]))
            tm.step({f"t{i}"}, [ent], dt=0.1)
        patches_before = sum(
            1 for p in self.plot._ax.patches if isinstance(p, mp.Ellipse)
        )
        self.plot._draw_track_ellipses(tm)
        ellipses_after = sum(
            1 for p in self.plot._ax.patches if isinstance(p, mp.Ellipse)
        )
        self.assertEqual(ellipses_after - patches_before, 3)

    def test_render_accepts_track_manager(self) -> None:
        """NF-VIZ-006 M5: render() accepts and uses track_manager param."""
        from sim3dves.entities.pedestrian import PedestrianEntity
        tm = TrackManager()
        ped = PedestrianEntity(
            entity_id="ped-r",
            position=np.array([100.0, 100.0, 0.0]),
            velocity=np.zeros(3),
        )
        ent = _make_entity("ped-r")
        tm.step({"ped-r"}, [ent], dt=0.1)
        # Should not raise
        self.plot.render([ped], sim_time=0.0, track_manager=tm)

    def test_track_panel_shows_quality(self) -> None:
        """NF-VIZ-006 M5: panel text includes track quality for tracked entity."""
        from sim3dves.entities.pedestrian import PedestrianEntity
        tm = _make_tm()
        ent_mock = _make_entity("ped-panel")
        for _ in range(_D.TRK_HIGH_HIT_COUNT + 1):
            tm.step({"ped-panel"}, [ent_mock], dt=0.1)
        # Attach track_manager as _last_track_manager
        self.plot._last_track_manager = tm
        ped = PedestrianEntity(
            entity_id="ped-panel",
            position=np.array([100.0, 100.0, 0.0]),
            velocity=np.zeros(3),
        )
        text = self.plot._build_panel_text(ped)
        self.assertIn("TrkQual", text)
        self.assertIn("HIGH", text)


# ---------------------------------------------------------------------------
# 10. defaults.py M5 constants
# ---------------------------------------------------------------------------

class TestM5Defaults(unittest.TestCase):
    """NF-M-006: all M5 constants are present and sane."""

    def test_process_noise_positive(self) -> None:
        """M5: TRK_PROCESS_NOISE_STD > 0."""
        self.assertGreater(_D.TRK_PROCESS_NOISE_STD, 0.0)

    def test_meas_noise_positive(self) -> None:
        """M5: TRK_MEAS_NOISE_STD > 0."""
        self.assertGreater(_D.TRK_MEAS_NOISE_STD, 0.0)

    def test_max_miss_steps_positive(self) -> None:
        """M5: TRK_MAX_MISS_STEPS >= 1."""
        self.assertGreaterEqual(_D.TRK_MAX_MISS_STEPS, 1)

    def test_quality_thresholds_ordered(self) -> None:
        """M5: MEDIUM threshold < HIGH threshold."""
        self.assertLess(_D.TRK_MEDIUM_HIT_COUNT, _D.TRK_HIGH_HIT_COUNT)

    def test_ellipse_alpha_in_unit_interval(self) -> None:
        """NF-VIZ-006 M5: TRK_VIZ_ELLIPSE_ALPHA in (0, 1]."""
        self.assertGreater(_D.TRK_VIZ_ELLIPSE_ALPHA, 0.0)
        self.assertLessEqual(_D.TRK_VIZ_ELLIPSE_ALPHA, 1.0)

    def test_colour_strings_nonempty(self) -> None:
        """NF-VIZ-006 M5: all track colour strings are non-empty."""
        for attr in ("TRK_VIZ_HIGH_COLOUR", "TRK_VIZ_MEDIUM_COLOUR",
                     "TRK_VIZ_LOW_COLOUR"):
            self.assertTrue(len(getattr(_D, attr)) > 0, f"{attr} is empty")


if __name__ == "__main__":
    unittest.main()
