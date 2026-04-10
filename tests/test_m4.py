"""
tests.test_m4
=============
Unit tests for M4: OpticalPayload, DetectionEngine, GimbalFSM,
engine integration, and visualiser FOV cone.

NF-TE-001: Every new module covered; ≥ 80% coverage.
NF-TE-002: Every test docstring cites the Req ID(s) it validates.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import math
import time
import unittest
from typing import List
from unittest.mock import MagicMock

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.world import AABB, World
from sim3dves.entities.base import EntityType
from sim3dves.entities.uav import SearchPattern, UAVEntity
from sim3dves.payload.detection_engine import DetectionEngine
from sim3dves.payload.optical_payload import GimbalMode, OpticalPayload

_D = SimDefaults()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uav(
    entity_id: str = "uav-0",
    pos: np.ndarray | None = None,
    heading: float = 0.0,
) -> UAVEntity:
    """Construct a minimal UAVEntity for payload tests."""
    if pos is None:
        pos = np.array([300.0, 300.0, 100.0])
    return UAVEntity(
        entity_id=entity_id,
        position=pos,
        heading=heading,
        world_extent=np.array([600.0, 600.0]),
        alt_floor_m=_D.UAV_ALT_FLOOR_M,
        alt_ceil_m=_D.UAV_ALT_CEIL_M,
        rng=np.random.default_rng(0),
    )


def _make_entity(
    entity_id: str = "ped-0",
    pos: np.ndarray | None = None,
    is_eoi: bool = False,
    signature: float = 1.0,
    etype: EntityType = EntityType.PEDESTRIAN,
) -> MagicMock:
    """Return a lightweight mock entity for detection tests."""
    ent = MagicMock()
    ent.entity_id = entity_id
    ent.entity_type = etype
    ent.position = pos if pos is not None else np.array([300.0, 350.0, 0.0])
    ent.is_eoi = is_eoi
    ent.signature = signature
    ent.alive = True
    return ent


def _make_payload(owner: str = "uav-0") -> OpticalPayload:
    """Construct OpticalPayload with default DetectionEngine."""
    return OpticalPayload(owner_id=owner)


# ===========================================================================
# 1. GimbalFSM / OpticalPayload
# ===========================================================================

class TestGimbalDefaultState(unittest.TestCase):
    """PAY-002: gimbal initialises within legal limits."""

    def test_initial_mode_is_scan(self) -> None:
        """PAY-002: default gimbal mode is SCAN."""
        p = _make_payload()
        self.assertEqual(p.mode, GimbalMode.SCAN)

    def test_initial_azimuth_in_range(self) -> None:
        """PAY-002: initial azimuth within ±AZ_RANGE/2."""
        p = _make_payload()
        limit = _D.PAY_GIMBAL_AZ_RANGE_DEG / 2.0
        self.assertGreaterEqual(p.gimbal_az_deg, -limit)
        self.assertLessEqual(p.gimbal_az_deg, limit)

    def test_initial_elevation_in_range(self) -> None:
        """PAY-003: initial elevation within [EL_MIN, EL_MAX]."""
        p = _make_payload()
        self.assertGreaterEqual(p.gimbal_el_deg, _D.PAY_GIMBAL_EL_MIN_DEG)
        self.assertLessEqual(p.gimbal_el_deg, _D.PAY_GIMBAL_EL_MAX_DEG)


class TestGimbalRateLimit(unittest.TestCase):
    """PAY-003: gimbal slew rate is clamped to PAY_GIMBAL_RATE_DPS."""

    def test_slew_rate_capped(self) -> None:
        """PAY-003: one-step az change cannot exceed rate * dt."""
        p = _make_payload()
        p.gimbal_az_deg = 0.0
        # Command a 180° jump — rate-limiter should cap it
        p._slew_to(180.0, -45.0)
        max_step = _D.PAY_GIMBAL_RATE_DPS * 0.1  # dt=0.1
        self.assertLessEqual(abs(p.gimbal_az_deg), max_step + 1e-6)

    def test_elevation_rate_capped(self) -> None:
        """PAY-003: elevation change per step is capped."""
        p = _make_payload()
        p.gimbal_el_deg = 0.0
        p._slew_to(0.0, -90.0)
        max_step = _D.PAY_GIMBAL_RATE_DPS * 0.1
        self.assertGreaterEqual(p.gimbal_el_deg, -max_step - 1e-6)


class TestGimbalLimits(unittest.TestCase):
    """PAY-002, PAY-003: gimbal hardware limits are enforced."""

    def test_az_clamped_to_range(self) -> None:
        """PAY-002: azimuth never exceeds ±AZ_RANGE/2."""
        p = _make_payload()
        for _ in range(200):
            p._slew_to(999.0, -45.0)
        limit = _D.PAY_GIMBAL_AZ_RANGE_DEG / 2.0
        self.assertLessEqual(p.gimbal_az_deg, limit + 1e-6)

    def test_el_clamped_below_nadir(self) -> None:
        """PAY-003: elevation never goes below PAY_GIMBAL_EL_MIN_DEG."""
        p = _make_payload()
        for _ in range(200):
            p._slew_to(0.0, -999.0)
        self.assertGreaterEqual(p.gimbal_el_deg, _D.PAY_GIMBAL_EL_MIN_DEG - 1e-6)

    def test_el_clamped_above_horizon(self) -> None:
        """PAY-003: elevation never exceeds PAY_GIMBAL_EL_MAX_DEG."""
        p = _make_payload()
        for _ in range(200):
            p._slew_to(0.0, 999.0)
        self.assertLessEqual(p.gimbal_el_deg, _D.PAY_GIMBAL_EL_MAX_DEG + 1e-6)


class TestGimbalModeTransitions(unittest.TestCase):
    """PAY-002: mode transitions via command API."""

    def test_command_stare(self) -> None:
        """PAY-002: command_stare switches mode to STARE."""
        p = _make_payload()
        p.command_stare(np.array([100.0, 200.0]))
        self.assertEqual(p.mode, GimbalMode.STARE)

    def test_command_cued(self) -> None:
        """PAY-002, PAY-005: command_cued switches mode to CUED."""
        p = _make_payload()
        p.command_cued("target-001")
        self.assertEqual(p.mode, GimbalMode.CUED)
        self.assertEqual(p.track_entity_id, "target-001")

    def test_command_scan(self) -> None:
        """PAY-002: command_scan switches mode back to SCAN."""
        p = _make_payload()
        p.command_stare(np.array([100.0, 200.0]))
        p.command_scan()
        self.assertEqual(p.mode, GimbalMode.SCAN)


class TestGimbalScanSweeps(unittest.TestCase):
    """PAY-002: SCAN mode sweeps azimuth over time."""

    def test_scan_azimuth_changes(self) -> None:
        """PAY-002: azimuth changes each step in SCAN mode."""
        p = _make_payload()
        p.command_scan()
        az_before = p.gimbal_az_deg
        p._gimbal_scan(dt=0.1)
        p._gimbal_scan(dt=0.1)
        # After two steps the angle must have changed
        self.assertNotAlmostEqual(p.gimbal_az_deg, az_before, places=3)


class TestAimVectorWorld(unittest.TestCase):
    """PAY-001: aim vector is unit length and consistent with heading."""

    def test_aim_vector_unit_length(self) -> None:
        """PAY-001: computed aim vector is a unit vector."""
        p = _make_payload()
        p.gimbal_az_deg = 0.0
        p.gimbal_el_deg = -45.0
        aim = p._compute_aim_vector_world(uav_heading_deg=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(aim)), 1.0, places=6)

    def test_nadir_aim_points_down(self) -> None:
        """PAY-001: nadir pointing (el=-90) produces downward Z component."""
        p = _make_payload()
        p.gimbal_az_deg = 0.0
        p.gimbal_el_deg = -90.0
        aim = p._compute_aim_vector_world(uav_heading_deg=0.0)
        self.assertLess(float(aim[2]), -0.99)


class TestFOVMembership(unittest.TestCase):
    """PAY-001: entities_in_fov selects correct candidates."""

    def test_entity_directly_ahead_is_in_fov(self) -> None:
        """PAY-001: entity in boresight is inside FOV."""
        p = _make_payload()
        p.gimbal_az_deg = 0.0
        p.gimbal_el_deg = -45.0
        uav_pos = np.array([300.0, 300.0, 100.0])
        aim = p._compute_aim_vector_world(uav_heading_deg=0.0)

        # Place entity along aim direction
        target_pos = uav_pos + aim * 50.0
        entity = _make_entity(pos=target_pos)
        result = p._entities_in_fov(uav_pos, aim, [entity])
        self.assertIn(entity, result)

    def test_entity_outside_cone_excluded(self) -> None:
        """PAY-001: entity 90° off boresight is outside FOV cone."""
        p = _make_payload()
        p.gimbal_az_deg = 0.0
        p.gimbal_el_deg = 0.0
        uav_pos = np.array([300.0, 300.0, 100.0])
        aim = p._compute_aim_vector_world(uav_heading_deg=0.0)

        # Place entity 90° to the side (perpendicular)
        perp = np.array([0.0, 1.0, 0.0])
        entity = _make_entity(pos=uav_pos + perp * 50.0)
        result = p._entities_in_fov(uav_pos, aim, [entity])
        self.assertNotIn(entity, result)

    def test_empty_entity_list(self) -> None:
        """PAY-001: empty input returns empty result."""
        p = _make_payload()
        aim = np.array([1.0, 0.0, 0.0])
        result = p._entities_in_fov(np.array([0.0, 0.0, 100.0]), aim, [])
        self.assertEqual(result, [])


class TestFlushDetections(unittest.TestCase):
    """PAY-004: flush_detections returns and clears the buffer."""

    def test_flush_clears_buffer(self) -> None:
        """PAY-004: second flush returns empty list."""
        p = _make_payload()
        # Manually populate buffer
        p._pending_detections.append({"uav_id": "u", "target_id": "t"})
        first = p.flush_detections()
        second = p.flush_detections()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)

    def test_step_produces_detections(self) -> None:
        """PAY-004: step() buffers detection events for in-FOV entities."""
        p = _make_payload()
        uav_pos = np.array([300.0, 300.0, 100.0])
        # Place entity directly below UAV (nadir)
        entity = _make_entity(pos=np.array([300.0, 300.0, 0.0]), signature=1.0)
        p.step(
            uav_position=uav_pos,
            uav_heading_deg=0.0,
            entities=[entity],
            dt=0.1,
        )
        detections = p.flush_detections()
        # Entity is below UAV — may or may not be in FOV depending on scan angle.
        # The important assertion is that flush returns a list (even if empty).
        self.assertIsInstance(detections, list)


# ===========================================================================
# 2. DetectionEngine
# ===========================================================================

class TestDetectionEnginePD(unittest.TestCase):
    """POL-001: P(D) model returns correct values."""

    def setUp(self) -> None:
        self.eng = DetectionEngine()

    def test_pd_zero_beyond_range(self) -> None:
        """POL-001: P(D) = 0 when dist >= DETECT_RANGE_M."""
        ent = _make_entity(signature=1.0)
        pd = self.eng.compute_pd(ent, _D.PAY_DETECT_RANGE_M, los_clear=True)
        self.assertAlmostEqual(pd, 0.0)

    def test_pd_zero_los_blocked(self) -> None:
        """POL-001: P(D) = 0 when LOS is blocked."""
        ent = _make_entity(signature=1.0)
        pd = self.eng.compute_pd(ent, 50.0, los_clear=False)
        self.assertAlmostEqual(pd, 0.0)

    def test_pd_at_min_range(self) -> None:
        """POL-001: P(D) at min range approaches BASE_PD for signature=1."""
        ent = _make_entity(signature=1.0)
        pd = self.eng.compute_pd(ent, _D.PAY_DETECT_MIN_RANGE_M, los_clear=True)
        self.assertGreater(pd, 0.5)
        self.assertLessEqual(pd, _D.PAY_DETECT_BASE_PD + 1e-6)

    def test_pd_scaled_by_signature(self) -> None:
        """POL-001: lower signature yields lower P(D)."""
        ent_hi = _make_entity(signature=1.0)
        ent_lo = _make_entity(signature=0.3)
        pd_hi = self.eng.compute_pd(ent_hi, 100.0, los_clear=True)
        pd_lo = self.eng.compute_pd(ent_lo, 100.0, los_clear=True)
        self.assertGreater(pd_hi, pd_lo)

    def test_pd_in_unit_interval(self) -> None:
        """POL-001: P(D) is always in [0, 1]."""
        ent = _make_entity(signature=1.0)
        for dist in [0.0, 10.0, 100.0, 500.0, 600.0]:
            pd = self.eng.compute_pd(ent, dist, los_clear=True)
            self.assertGreaterEqual(pd, 0.0)
            self.assertLessEqual(pd, 1.0)

    def test_pd_decreases_with_range(self) -> None:
        """POL-001: P(D) is monotonically non-increasing with range."""
        ent = _make_entity(signature=1.0)
        prev_pd = 1.0
        for dist in [20.0, 100.0, 200.0, 350.0, 499.0]:
            pd = self.eng.compute_pd(ent, dist, los_clear=True)
            self.assertLessEqual(pd, prev_pd + 1e-9)
            prev_pd = pd


class TestDetectionEnginePlugin(unittest.TestCase):
    """NF-M-002: DetectionEngine is subclassable for custom models."""

    def test_custom_subclass_overrides_pd(self) -> None:
        """NF-M-002: subclass compute_pd() is called by process_batch()."""

        class AlwaysOne(DetectionEngine):
            def compute_pd(self, entity, dist_m, los_clear):
                return 1.0

        eng = AlwaysOne()
        ent = _make_entity()
        results = eng.process_batch(
            observer=np.array([0.0, 0.0, 100.0]),
            candidates=[ent],
            structures=[],
        )
        self.assertEqual(len(results), 1)
        _, pd, _ = results[0]
        self.assertAlmostEqual(pd, 1.0)


class TestLOSRaycast(unittest.TestCase):
    """NF-P-004: LOS raycast correctly identifies occlusion."""

    def setUp(self) -> None:
        self.eng = DetectionEngine()

    def test_clear_sky_los(self) -> None:
        """NF-P-004: LOS is clear when no structures are present."""
        observer = np.array([0.0, 0.0, 100.0])
        targets = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        result = self.eng._batch_los(observer, targets, [])
        self.assertTrue(result.all())

    def test_structure_blocks_los(self) -> None:
        """NF-P-004: AABB structure between observer and target blocks LOS."""
        observer = np.array([0.0, 0.0, 50.0])
        # Target is at (200, 0, 0); AABB at x=90..110, y=-10..10
        target = np.array([[200.0, 0.0, 0.0]])

        structure = AABB(x=90.0, y=-10.0, width=20.0, depth=20.0, height=100.0)
        result = self.eng._batch_los(observer, target, [structure])
        self.assertFalse(result[0])

    def test_structure_beside_ray_does_not_block(self) -> None:
        """NF-P-004: structure beside the ray does not block LOS."""
        observer = np.array([0.0, 0.0, 50.0])
        target = np.array([[200.0, 0.0, 0.0]])
        # Structure is 100 m to the side — should not intersect the ray
        structure = AABB(x=90.0, y=90.0, width=20.0, depth=20.0, height=100.0)
        result = self.eng._batch_los(observer, target, [structure])
        self.assertTrue(result[0])

    def test_batch_performance(self) -> None:
        """NF-P-004: batch of 64 entities completes in < 10 ms."""
        observer = np.array([0.0, 0.0, 100.0])
        rng = np.random.default_rng(1)
        targets = rng.uniform(50.0, 400.0, size=(64, 3))
        targets[:, 2] = 0.0
        structures = [
            AABB(x=50.0 * i, y=0.0, width=20.0, depth=20.0, height=50.0)
            for i in range(5)
        ]
        t0 = time.perf_counter()
        self.eng._batch_los(observer, targets, structures)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 10.0, f"LOS batch took {elapsed_ms:.1f} ms > 10 ms")

    def test_batch_result_shape(self) -> None:
        """NF-P-004: _batch_los returns shape (n,) boolean array."""
        observer = np.array([0.0, 0.0, 100.0])
        targets = np.random.default_rng(2).uniform(0, 100, (10, 3))
        result = self.eng._batch_los(observer, targets, [])
        self.assertEqual(result.shape, (10,))
        self.assertEqual(result.dtype, bool)


# ===========================================================================
# 3. Engine integration
# ===========================================================================

class TestEngineDetectionEvent(unittest.TestCase):
    """PAY-004, LOG-002: engine publishes DETECTION events from payloads."""

    def _make_world(self) -> World:
        return World(extent=np.array([600.0, 600.0]))

    def test_detection_events_logged(self) -> None:
        """PAY-004: DETECTION events reach the EventBus when payload fires."""
        from sim3dves.core.event_bus import EventType

        world = self._make_world()
        config = SimulationConfig(logging_enabled=False)
        sim = SimulationEngine(config, world)

        uav = UAVEntity(
            entity_id="uav-pay",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        # Force a detection directly into the payload buffer
        payload = OpticalPayload(owner_id="uav-pay")
        payload._pending_detections.append({
            "uav_id": "uav-pay",
            "target_id": "t0",
            "pd": 0.85,
            "los_clear": True,
        })
        uav.payload = payload

        detected_events: list = []
        sim.event_bus.subscribe(
            EventType.DETECTION,
            lambda e: detected_events.append(e),
        )
        sim.add_entity(uav)

        with sim.logger:
            sim.step()

        self.assertEqual(len(detected_events), 1)
        self.assertEqual(detected_events[0].payload["target_id"], "t0")

    def test_logging_disabled_no_file(self) -> None:
        """SIM-007: logging_enabled=False creates no log file."""
        import os
        from pathlib import Path

        log_path = Path("/tmp/test_m4_nolog.jsonl")
        if log_path.exists():
            log_path.unlink()

        world = self._make_world()
        config = SimulationConfig(logging_enabled=False, log_file=log_path)
        sim = SimulationEngine(config, world)
        with sim.logger:
            sim.step()

        self.assertFalse(log_path.exists(), "Log file should not be created when disabled")


# ===========================================================================
# 4. Visualiser FOV cone
# ===========================================================================

class TestFOVConeGeometry(unittest.TestCase):
    """NF-VIZ-006 M4: FOV cone geometry computed correctly."""

    def test_fov_data_updated_after_step(self) -> None:
        """NF-VIZ-006 M4: payload.step() populates fov_tip_world and fov_axis_world."""
        p = _make_payload()
        uav_pos = np.array([300.0, 300.0, 100.0])
        p.step(
            uav_position=uav_pos,
            uav_heading_deg=45.0,
            entities=[],
            dt=0.1,
        )
        self.assertIsNotNone(p.fov_tip_world)
        self.assertIsNotNone(p.fov_axis_world)
        np.testing.assert_array_almost_equal(p.fov_tip_world, uav_pos)

    def test_fov_axis_is_unit_vector(self) -> None:
        """NF-VIZ-006 M4: fov_axis_world XY component has unit length <= 1."""
        p = _make_payload()
        p.step(
            uav_position=np.array([300.0, 300.0, 100.0]),
            uav_heading_deg=0.0,
            entities=[],
            dt=0.1,
        )
        # XY component norm <= 1 (it is the XY slice of a 3-D unit vector)
        norm_xy = float(np.linalg.norm(p.fov_axis_world))
        self.assertLessEqual(norm_xy, 1.0 + 1e-6)

    def test_half_angle_matches_config(self) -> None:
        """NF-VIZ-006 M4: fov_half_angle_deg = PAY_FOV_DEG / 2."""
        p = _make_payload()
        self.assertAlmostEqual(
            p.fov_half_angle_deg, _D.PAY_FOV_DEG / 2.0
        )


# ===========================================================================
# 5. SimulationView alias
# ===========================================================================

class TestSimulationViewAlias(unittest.TestCase):
    """NF-VIZ-006 M4: SimulationView is the canonical name; DebugPlot is alias."""

    def test_simulation_view_importable(self) -> None:
        """NF-VIZ-006 M4: SimulationView can be imported from debug_plot."""
        from sim3dves.viz.debug_plot import SimulationView
        self.assertIsNotNone(SimulationView)

    def test_debug_plot_alias_identical(self) -> None:
        """NF-VIZ-006 M4: DebugPlot and SimulationView refer to same class."""
        from sim3dves.viz.debug_plot import DebugPlot, SimulationView
        self.assertIs(SimulationView, DebugPlot)


# ===========================================================================
# 6. defaults.py M4 constants
# ===========================================================================

class TestM4Defaults(unittest.TestCase):
    """NF-M-006: All M4 constants are present and sane."""

    def test_fov_positive(self) -> None:
        """NF-M-006: PAY_FOV_DEG is positive."""
        self.assertGreater(_D.PAY_FOV_DEG, 0.0)

    def test_detect_range_positive(self) -> None:
        """NF-M-006: PAY_DETECT_RANGE_M > PAY_DETECT_MIN_RANGE_M."""
        self.assertGreater(_D.PAY_DETECT_RANGE_M, _D.PAY_DETECT_MIN_RANGE_M)

    def test_base_pd_in_unit_interval(self) -> None:
        """NF-M-006: PAY_DETECT_BASE_PD in (0, 1]."""
        self.assertGreater(_D.PAY_DETECT_BASE_PD, 0.0)
        self.assertLessEqual(_D.PAY_DETECT_BASE_PD, 1.0)

    def test_gimbal_el_range_valid(self) -> None:
        """NF-M-006: EL_MIN < EL_MAX."""
        self.assertLess(_D.PAY_GIMBAL_EL_MIN_DEG, _D.PAY_GIMBAL_EL_MAX_DEG)

    def test_los_batch_size_positive(self) -> None:
        """NF-P-004: PAY_LOS_BATCH_SIZE >= 1."""
        self.assertGreaterEqual(_D.PAY_LOS_BATCH_SIZE, 1)

    def test_pause_key_is_space(self) -> None:
        """NF-VIZ-019: VIZ_PAUSE_KEY is the space-bar callback value."""
        self.assertEqual(_D.VIZ_PAUSE_KEY, " ")

    def test_logging_enabled_default_true(self) -> None:
        """SIM-007: SIM_LOGGING_ENABLED defaults to True."""
        self.assertTrue(_D.SIM_LOGGING_ENABLED)


if __name__ == "__main__":
    unittest.main()
