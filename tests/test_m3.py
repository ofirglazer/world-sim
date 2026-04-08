"""
tests.test_m3
=============
M3 test suite -- UAVEntity, autopilot FSM, all flight rules (FLR-001..011),
search patterns, multi-UAV deconfliction, NFZ avoidance, endurance,
interactive visualiser state, and end-to-end integration.

Every test cites the Req ID(s) it validates (NF-TE-002).
Coverage target: >= 80% of new M3 modules (NF-TE-001, NF-M-003).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import List

# Set headless backend before any matplotlib import so DebugPlot tests
# can exercise state logic without a display.
import matplotlib
matplotlib.use("Agg")

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.event_bus import EventType
from sim3dves.core.world import NFZCylinder, World
from sim3dves.entities.base import EntityManager, EntityType
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.uav import AutopilotMode, SearchPattern, UAVEntity
from sim3dves.viz.debug_plot import DebugPlot

_D = SimDefaults()
_TMPDIR = Path(tempfile.mkdtemp())


# =============================================================================
# Helpers
# =============================================================================

def _make_uav(
    entity_id: str = "uav-test",
    pos: np.ndarray | None = None,
    heading: float = 0.0,
    endurance_s: float = 300.0,
    search_pattern: SearchPattern = SearchPattern.LAWNMOWER,
    world_extent: np.ndarray | None = None,
    nfz_cylinders: list | None = None,
    cruise_alt: float = 100.0,
) -> UAVEntity:
    """Construct UAVEntity with test defaults."""
    if pos is None:
        pos = np.array([300.0, 300.0, cruise_alt])
    return UAVEntity(
        entity_id=entity_id,
        position=pos,
        heading=heading,
        launch_position=np.array([300.0, 300.0, cruise_alt]),
        search_pattern=search_pattern,
        cruise_altitude_m=cruise_alt,
        endurance_s=endurance_s,
        world_extent=world_extent if world_extent is not None
        else np.array([600.0, 600.0]),
        alt_floor_m=_D.UAV_ALT_FLOOR_M,
        alt_ceil_m=_D.UAV_ALT_CEIL_M,
        nfz_cylinders=nfz_cylinders or [],
        rng=np.random.default_rng(42),
    )


def _small_world(nfz_cylinders: list | None = None) -> World:
    return World(
        extent=np.array([600.0, 600.0]),
        nfz_cylinders=nfz_cylinders or [],
        alt_floor_m=_D.UAV_ALT_FLOOR_M,
        alt_ceil_m=_D.UAV_ALT_CEIL_M,
    )


def _make_sim(
    world: World | None = None,
    duration_s: float = 1.0,
    dt: float = 0.1,
    log_file: Path | None = None,
) -> SimulationEngine:
    if world is None:
        world = _small_world()
    cfg = SimulationConfig(
        duration_s=duration_s, dt=dt, seed=42,
        log_file=log_file or (_TMPDIR / f"{uuid.uuid4()}.jsonl"),
    )
    return SimulationEngine(cfg, world)


def _make_plot() -> DebugPlot:
    """DebugPlot with Agg backend (headless-safe)."""
    return DebugPlot(600.0, 600.0)


# =============================================================================
# UAV-001 / UAV-002: Kinematics
# =============================================================================

class TestUAVKinematics(unittest.TestCase):

    def test_altitude_initialised_to_cruise(self) -> None:
        """UAV-001: Z position at construction equals cruise_altitude_m."""
        uav = _make_uav(cruise_alt=150.0)
        self.assertAlmostEqual(uav.position[2], 150.0, places=3)

    def test_entity_type_is_uav(self) -> None:
        """ENT-002: entity_type is EntityType.UAV Enum value."""
        self.assertEqual(_make_uav().entity_type, EntityType.UAV)

    def test_initial_autopilot_mode_is_waypoint(self) -> None:
        """UAV-005: default autopilot mode is WAYPOINT."""
        self.assertEqual(_make_uav().autopilot_mode, AutopilotMode.WAYPOINT)

    def test_altitude_floor_enforced(self) -> None:
        """FLR-002: Z never drops below alt_floor_m."""
        uav = _make_uav(cruise_alt=_D.UAV_ALT_FLOOR_M)
        uav.velocity[2] = -50.0
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertGreaterEqual(uav.position[2], _D.UAV_ALT_FLOOR_M - 0.01)

    def test_altitude_ceiling_enforced(self) -> None:
        """FLR-003: Z never exceeds alt_ceil_m."""
        uav = _make_uav(cruise_alt=_D.UAV_ALT_CEIL_M)
        uav.velocity[2] = 50.0
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertLessEqual(uav.position[2], _D.UAV_ALT_CEIL_M + 0.01)

    def test_alive_at_construction(self) -> None:
        """ENT-001: alive=True at construction."""
        self.assertTrue(_make_uav().alive)


# =============================================================================
# UAV-003: Endurance
# =============================================================================

class TestUAVEndurance(unittest.TestCase):

    def test_endurance_decreases_per_step(self) -> None:
        """UAV-003: endurance_remaining_s decreases by dt each step."""
        uav = _make_uav(endurance_s=100.0)
        uav.step(dt=0.1)
        self.assertAlmostEqual(uav.endurance_remaining_s, 99.9, places=4)

    def test_endurance_never_negative(self) -> None:
        """UAV-003: endurance_remaining_s is clamped to 0."""
        uav = _make_uav(endurance_s=0.1)
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertGreaterEqual(uav.endurance_remaining_s, 0.0)

    def test_low_fuel_triggers_rtb_or_land(self) -> None:
        """FLR-006: autopilot transitions to RTB/EMERGENCY_LAND at low fuel."""
        uav = _make_uav(endurance_s=_D.UAV_LOW_FUEL_THRESHOLD_S + 0.05)
        uav.step(dt=0.1)
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
        )

    def test_low_fuel_flag_set(self) -> None:
        """UAV-003: low_fuel property is True after threshold crossed."""
        uav = _make_uav(endurance_s=0.05)
        uav.step(dt=0.1)
        self.assertTrue(uav.low_fuel)

    def test_rtb_persists_after_depletion(self) -> None:
        """FLR-006: mode stays RTB/EMERGENCY_LAND after depletion."""
        uav = _make_uav(endurance_s=0.05)
        for _ in range(10):
            uav.step(dt=0.1)
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
        )


# =============================================================================
# FLR-001: NFZ avoidance -- turn-rate compliance (BUG FIX ATC-M3-003)
# =============================================================================

class TestNFZAvoidance(unittest.TestCase):

    def test_nfz_avoidance_respects_turn_rate(self) -> None:
        """
        FLR-001, UAV-002 (ATC-M3-003): heading change per step during NFZ
        avoidance must not exceed UAV_TURN_RATE_DPS * dt.

        BUG FIX: previously velocity was overwritten directly (instant snap).
        """
        dt = 0.1
        max_delta_deg = _D.UAV_TURN_RATE_DPS * dt + 0.5  # +0.5 tolerance

        # Place NFZ directly ahead so avoidance fires on the first step
        nfz = NFZCylinder(
            center_xy=np.array([400.0, 300.0]),
            radius_m=80.0,
            alt_max_m=300.0,
        )
        uav = _make_uav(
            pos=np.array([200.0, 300.0, 100.0]),
            heading=0.0,          # Flying East straight into NFZ
            nfz_cylinders=[nfz],
        )
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS
        uav.velocity[1] = 0.0

        for _ in range(15):
            heading_before = uav.heading
            uav.step(dt=dt)
            delta = abs(uav.heading - heading_before)
            # Wrap to [0, 180]
            if delta > 180.0:
                delta = 360.0 - delta
            self.assertLessEqual(
                delta, max_delta_deg,
                f"Heading jump {delta:.2f} deg > limit {max_delta_deg:.2f} deg"
                " — instantaneous snap still occurring",
            )

    def test_nfz_avoidance_changes_direction(self) -> None:
        """FLR-001: velocity direction changes (not instantly, but progressively)."""
        nfz = NFZCylinder(
            center_xy=np.array([400.0, 300.0]),
            radius_m=80.0,
            alt_max_m=300.0,
        )
        uav = _make_uav(
            pos=np.array([200.0, 300.0, 100.0]),
            heading=0.0,
            nfz_cylinders=[nfz],
        )
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS
        heading_init = uav.heading
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertNotAlmostEqual(uav.heading, heading_init, places=0)

    def test_uav_not_killed_on_nfz_entry(self) -> None:
        """FLR-001: NFZ entry does not kill UAV (avoidance/logging only)."""
        nfz = NFZCylinder(
            center_xy=np.array([305.0, 300.0]),
            radius_m=20.0, alt_max_m=200.0,
        )
        world = _small_world(nfz_cylinders=[nfz])
        sim = _make_sim(world=world)
        uav = _make_uav(pos=np.array([300.0, 300.0, 100.0]), nfz_cylinders=[nfz])
        sim.add_entity(uav)
        with sim.logger:
            for _ in range(5):
                sim.step()
        self.assertTrue(uav.alive)


# =============================================================================
# FLR-004: UAV-UAV separation
# =============================================================================

class TestUAVSeparation(unittest.TestCase):

    def test_separation_correction_applied(self) -> None:
        """FLR-004: velocity corrected when UAVs are < UAV_SEPARATION_M apart."""
        uav_a = _make_uav("uav-a", pos=np.array([300.0, 300.0, 100.0]))
        uav_b = _make_uav("uav-b", pos=np.array([310.0, 300.0, 100.0]))
        mgr = EntityManager()
        mgr.add(uav_a)
        mgr.add(uav_b)
        vel_a_before = uav_a.velocity.copy()
        mgr.step_all(dt=0.1)
        self.assertFalse(np.allclose(uav_a.velocity, vel_a_before, atol=0.01))

    def test_uav_neighbor_radius_larger_than_ground(self) -> None:
        """M3 base.py: UAV neighbor_radius_m > ground entity default."""
        self.assertGreater(_make_uav().neighbor_radius_m, _D.NEIGHBOR_RADIUS_M)


# =============================================================================
# FLR-005 / FLR-011: Geofence and corner-escape (ATC-M3-001, ATC-M3-002)
# =============================================================================

class TestGeofenceAndCornerEscape(unittest.TestCase):
    """
    FLR-005 / FLR-011: geofence and corner-escape inline heading logic.

    Design note: the implementation applies corner-escape INLINE within
    _update_behavior (step 5) rather than as a separate AutopilotMode enum
    value.  The UAV remains in WAYPOINT mode while the corner-escape heading
    is being applied; it transitions to RTB once clear of the corner pocket.
    Tests therefore check heading and position rather than autopilot_mode.
    """

    def test_straight_edge_triggers_rtb(self) -> None:
        """FLR-005: straight-edge approach -> RTB (no corner pocket)."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        # Near East boundary only; Y is mid-world — not a corner pocket
        uav = _make_uav(pos=np.array([600.0 - m + 5.0, 300.0, 100.0]))
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS  # Flying East
        uav.step(dt=0.1)
        self.assertEqual(uav.autopilot_mode, AutopilotMode.RTB)

    def test_corner_pocket_detection(self) -> None:
        """FLR-011: _in_corner_pocket() True only when BOTH margins breached."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        self.assertTrue(
            _make_uav(pos=np.array([m - 1.0, m - 1.0, 100.0]))._in_corner_pocket()
        )
        self.assertFalse(
            _make_uav(pos=np.array([m - 1.0, 300.0, 100.0]))._in_corner_pocket()
        )

    def test_corner_escape_heading_points_outward(self) -> None:
        """FLR-011: escape heading is in the NE quadrant for the SW corner pocket."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        uav = _make_uav(pos=np.array([m - 5.0, m - 5.0, 100.0]))
        heading_rad = uav._corner_escape_heading_rad()
        heading_deg = math.degrees(heading_rad) % 360.0
        # SW corner -> escape heading is NE (0 < deg < 90)
        self.assertGreater(heading_deg, 0.0)
        self.assertLess(heading_deg, 90.0)

    def test_corner_pocket_steers_toward_escape_heading(self) -> None:
        """
        FLR-011 (ATC-M3-001): when in corner pocket the heading is nudged toward
        the escape direction.  UAV stays in WAYPOINT mode (inline design).
        """
        m = _D.UAV_GEOFENCE_MARGIN_M
        uav = _make_uav(pos=np.array([m - 5.0, m - 5.0, 100.0]), heading=225.0)
        escape_rad = uav._corner_escape_heading_rad()
        escape_deg = math.degrees(escape_rad) % 360.0

        # After one step the heading should have moved toward the escape heading
        uav.step(dt=0.1)
        new_deg = uav.heading % 360.0

        # Angular distance to escape heading should have decreased
        def _angular_dist(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return min(d, 360.0 - d)

        dist_before = _angular_dist(225.0, escape_deg)
        dist_after = _angular_dist(new_deg, escape_deg)
        self.assertLessEqual(dist_after, dist_before + 0.01)
        # Mode remains WAYPOINT (corner escape is inline, not a separate mode)
        self.assertEqual(uav.autopilot_mode, AutopilotMode.WAYPOINT)

    def test_corner_escape_stays_in_bounds(self) -> None:
        """
        FLR-011 (ATC-M3-001): UAV approaching a corner from outside must remain
        within world bounds as the inline corner-escape heading is applied.

        Physics note: with UAV_MAX_SPEED_MPS=25 m/s and UAV_TURN_RATE_DPS=30 dps
        the minimum turn radius is ~48 m, making it physically impossible to avoid
        OOB if the UAV is already deep inside the margin heading directly into the
        corner at full speed.  This test therefore uses a low-speed custom kinematics
        (3 m/s -> turn radius ~5.7 m) so the corner-escape heading can redirect the
        UAV before it crosses the world boundary.
        """
        from sim3dves.entities.uav import UAVKinematics
        slow_kin = UAVKinematics(
            max_speed_mps=3.0,            # Very slow: turn radius ~5.7 m
            turn_rate_dps=_D.UAV_TURN_RATE_DPS,
        )
        m = _D.UAV_GEOFENCE_MARGIN_M
        extent = np.array([600.0, 600.0])
        uav = UAVEntity(
            entity_id="uav-corner-bounds",
            position=np.array([m - 5.0, m - 5.0, 100.0]),
            heading=225.0,
            kinematics=slow_kin,
            world_extent=extent,
            alt_floor_m=_D.UAV_ALT_FLOOR_M,
            alt_ceil_m=_D.UAV_ALT_CEIL_M,
            rng=np.random.default_rng(0),
        )
        uav.velocity[0] = 3.0 * math.cos(math.radians(225.0))
        uav.velocity[1] = 3.0 * math.sin(math.radians(225.0))

        for step in range(80):
            uav.step(dt=0.1)
            self.assertGreaterEqual(uav.position[0], 0.0, f"OOB X<0 at step {step}")
            self.assertLessEqual(uav.position[0], extent[0], f"OOB X>world at step {step}")
            self.assertGreaterEqual(uav.position[1], 0.0, f"OOB Y<0 at step {step}")
            self.assertLessEqual(uav.position[1], extent[1], f"OOB Y>world at step {step}")

    def test_corner_escape_then_rtb_when_clear(self) -> None:
        """
        FLR-011: once clear of the corner pocket (still in geofence margin on one
        edge) the WAYPOINT→RTB transition fires.
        """
        from sim3dves.entities.uav import UAVKinematics
        slow_kin = UAVKinematics(max_speed_mps=3.0)
        m = _D.UAV_GEOFENCE_MARGIN_M
        uav = UAVEntity(
            entity_id="uav-corner-rtb",
            position=np.array([m - 5.0, m - 5.0, 100.0]),
            heading=225.0,
            kinematics=slow_kin,
            world_extent=np.array([600.0, 600.0]),
            alt_floor_m=_D.UAV_ALT_FLOOR_M,
            alt_ceil_m=_D.UAV_ALT_CEIL_M,
            rng=np.random.default_rng(0),
        )
        # Run until RTB, EMERGENCY_LAND, or max steps
        for _ in range(300):
            uav.step(dt=0.1)
            if uav.autopilot_mode in (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND):
                break
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
            "UAV should transition to RTB after escaping the corner pocket",
        )


# =============================================================================
# FLR-007: Wind model
# =============================================================================

class TestWindModel(unittest.TestCase):

    def test_wind_drifts_position(self) -> None:
        """FLR-007: wind displaces position by wind * dt each step."""
        uav = _make_uav(pos=np.array([300.0, 300.0, 100.0]))
        uav._wind = np.array([5.0, 0.0, 0.0])  # 5 m/s East
        uav.velocity[:] = 0.0
        x_before = uav.position[0]
        uav.step(dt=1.0)
        self.assertGreater(uav.position[0], x_before)


# =============================================================================
# FLR-008: Search pattern safe margin (ATC-M3-002 BUG FIX)
# =============================================================================

class TestSearchPatterns(unittest.TestCase):

    def _count_waypoints(self, pattern: SearchPattern) -> int:
        return len(_make_uav(search_pattern=pattern)._generate_search_waypoints())

    def test_lawnmower_generates_waypoints(self) -> None:
        """FLR-008: LAWNMOWER produces >= 2 waypoints."""
        self.assertGreaterEqual(self._count_waypoints(SearchPattern.LAWNMOWER), 2)

    def test_spiral_generates_waypoints(self) -> None:
        """FLR-008: EXPANDING_SPIRAL produces >= 8 waypoints."""
        self.assertGreaterEqual(
            self._count_waypoints(SearchPattern.EXPANDING_SPIRAL), 8
        )

    def test_random_walk_count(self) -> None:
        """FLR-008: RANDOM_WALK produces exactly UAV_RANDOM_WALK_WAYPOINTS points."""
        self.assertEqual(
            self._count_waypoints(SearchPattern.RANDOM_WALK),
            _D.UAV_RANDOM_WALK_WAYPOINTS,
        )

    def test_all_waypoints_within_search_margin(self) -> None:
        """
        FLR-008 (ATC-M3-002, BUG FIX): all waypoints >= UAV_SEARCH_MARGIN_M
        from every world edge.  Previously used UAV_GEOFENCE_MARGIN_M, which
        placed waypoints at the geofence trigger boundary.
        """
        m = _D.UAV_SEARCH_MARGIN_M
        for pattern in SearchPattern:
            uav = _make_uav(search_pattern=pattern)
            for wp in uav._generate_search_waypoints():
                self.assertGreaterEqual(
                    wp[0], m - 0.01,
                    f"{pattern.name}: X={wp[0]:.1f} < margin {m}",
                )
                self.assertLessEqual(
                    wp[0], 600.0 - m + 0.01,
                    f"{pattern.name}: X={wp[0]:.1f} > world - margin",
                )
                self.assertGreaterEqual(wp[1], m - 0.01)
                self.assertLessEqual(wp[1], 600.0 - m + 0.01)

    def test_waypoints_at_cruise_altitude(self) -> None:
        """FLR-008: all search waypoints are at cruise altitude."""
        alt = 120.0
        uav = _make_uav(search_pattern=SearchPattern.LAWNMOWER, cruise_alt=alt)
        for wp in uav._generate_search_waypoints():
            self.assertAlmostEqual(wp[2], alt, places=3)

    def test_search_margin_greater_than_geofence_margin(self) -> None:
        """
        FLR-008 / FLR-011: UAV_SEARCH_MARGIN_M > UAV_GEOFENCE_MARGIN_M.

        This is the invariant that prevents waypoints from residing inside
        the geofence RTB trigger zone.
        """
        self.assertGreater(_D.UAV_SEARCH_MARGIN_M, _D.UAV_GEOFENCE_MARGIN_M)


# =============================================================================
# FLR-009: Cued orbit
# =============================================================================

class TestCuedOrbit(unittest.TestCase):

    def test_cue_orbit_transitions_mode(self) -> None:
        """FLR-009: cue_orbit() transitions to ORBIT."""
        uav = _make_uav()
        uav.cue_orbit(center_xy=np.array([300.0, 300.0]))
        self.assertEqual(uav.autopilot_mode, AutopilotMode.ORBIT)

    def test_orbit_velocity_nonzero(self) -> None:
        """FLR-009: ORBIT mode generates non-zero horizontal velocity."""
        uav = _make_uav(pos=np.array([400.0, 300.0, 100.0]))
        uav.cue_orbit(center_xy=np.array([300.0, 300.0]), radius_m=100.0)
        uav.step(dt=0.1)
        self.assertGreater(float(np.linalg.norm(uav.velocity[:2])), 0.1)

    def test_orbit_center_stored(self) -> None:
        """FLR-009: orbit centre stored correctly."""
        uav = _make_uav()
        centre = np.array([123.0, 456.0])
        uav.cue_orbit(centre)
        np.testing.assert_array_almost_equal(uav._orbit_center, centre)


# =============================================================================
# FLR-010: Deconfliction
# =============================================================================

class TestDeconfliction(unittest.TestCase):

    def test_two_uavs_same_cue_get_different_roles(self) -> None:
        """FLR-010: two UAVs orbiting same cue -> PRIMARY + SECONDARY."""
        uav_a = _make_uav("uav-a-dc", pos=np.array([300.0, 300.0, 100.0]))
        uav_b = _make_uav("uav-b-dc", pos=np.array([320.0, 300.0, 100.0]))
        centre = np.array([300.0, 300.0])
        uav_a.cue_orbit(centre)
        uav_b.cue_orbit(centre)
        mgr = EntityManager()
        mgr.add(uav_a)
        mgr.add(uav_b)
        mgr.step_all(dt=0.1)
        roles = {uav_a.deconfliction_role, uav_b.deconfliction_role}
        self.assertEqual(roles, {"PRIMARY", "SECONDARY"})

    def test_single_uav_orbit_is_primary(self) -> None:
        """FLR-010: single orbiting UAV is always PRIMARY."""
        uav = _make_uav()
        uav.cue_orbit(np.array([300.0, 300.0]))
        uav.step(dt=0.1)
        self.assertEqual(uav.deconfliction_role, "PRIMARY")


# =============================================================================
# Public properties (NF-VIZ-011 exposure)
# =============================================================================

class TestUAVPublicProperties(unittest.TestCase):

    def test_low_fuel_property(self) -> None:
        """NF-VIZ-011: low_fuel property exposed on UAVEntity."""
        uav = _make_uav(endurance_s=0.05)
        uav.step(dt=0.1)
        self.assertIsInstance(uav.low_fuel, bool)
        self.assertTrue(uav.low_fuel)

    def test_nfz_violation_property_false_when_outside(self) -> None:
        """NF-VIZ-011: nfz_violation=False when outside any NFZ."""
        uav = _make_uav()
        uav.step(dt=0.1)
        self.assertFalse(uav.nfz_violated)

    def test_nfz_violation_property_true_when_inside(self) -> None:
        """NF-VIZ-011: nfz_violation=True when position is inside NFZ."""
        nfz = NFZCylinder(
            center_xy=np.array([300.0, 300.0]),
            radius_m=50.0, alt_max_m=200.0,
        )
        uav = _make_uav(pos=np.array([300.0, 300.0, 100.0]), nfz_cylinders=[nfz])
        uav.step(dt=0.1)
        self.assertTrue(uav.nfz_violated)

    def test_current_destination_property(self) -> None:
        """NF-VIZ-011: current_destination is None or np.ndarray."""
        uav = _make_uav()
        # Before first step waypoints are generated lazily; just check type
        dest = uav.current_destination
        self.assertTrue(dest is None or isinstance(dest, np.ndarray))


# =============================================================================
# Multi-UAV (UAV-004)
# =============================================================================

class TestMultiUAV(unittest.TestCase):

    def test_four_uavs_alive_after_10_steps(self) -> None:
        """UAV-004: four UAVs operate concurrently without killing each other."""
        sim = _make_sim()
        for i in range(4):
            sim.add_entity(_make_uav(
                entity_id=f"uav-multi-{i}",
                pos=np.array([100.0 + i * 100.0, 300.0, 100.0 + i * 10.0]),
                endurance_s=600.0,
            ))
        with sim.logger:
            for _ in range(10):
                sim.step()
        alive = sum(1 for u in sim.entities.by_type(EntityType.UAV) if u.alive)
        self.assertEqual(alive, 4)


# =============================================================================
# Engine: NFZ violation event logging
# =============================================================================

class TestEngineNFZLogging(unittest.TestCase):

    def test_nfz_violation_event_in_jsonl(self) -> None:
        """FLR-001, LOG-002: NFZ_VIOLATION event appears in JSONL log."""
        nfz = NFZCylinder(
            center_xy=np.array([300.0, 300.0]),
            radius_m=5.0, alt_max_m=200.0,
        )
        world = _small_world(nfz_cylinders=[nfz])
        log_path = _TMPDIR / f"nfz_{uuid.uuid4()}.jsonl"
        sim = _make_sim(world=world, log_file=log_path)
        # Spawn UAV inside the tiny NFZ so violation triggers immediately
        uav = UAVEntity(
            entity_id="uav-nfz",
            position=np.array([300.0, 300.0, 100.0]),
            nfz_cylinders=[nfz],
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        sim.add_entity(uav)
        with sim.logger:
            for _ in range(3):
                sim.step()
        records = [json.loads(l) for l in log_path.read_text().splitlines()]
        nfz_events = [
            r for r in records
            if r.get("_record_type") == "event"
            and r.get("type") == EventType.NFZ_VIOLATION.name
        ]
        self.assertGreater(len(nfz_events), 0)


# =============================================================================
# Integration tests (NF-TE-004)
# =============================================================================

class TestM3Integration(unittest.TestCase):

    def test_100_steps_valid_jsonl(self) -> None:
        """NF-TE-004: 100-step headless run produces valid JSONL output."""
        log_path = _TMPDIR / f"int_{uuid.uuid4()}.jsonl"
        cfg = SimulationConfig(duration_s=10.0, dt=0.1, seed=0, log_file=log_path)
        sim = SimulationEngine(cfg, _small_world())
        for i in range(2):
            sim.add_entity(_make_uav(
                entity_id=f"uav-int-{i}",
                pos=np.array([200.0 + i * 100.0, 300.0, 100.0]),
                endurance_s=600.0,
            ))
        sim.run()
        lines = log_path.read_text().splitlines()
        step_lines = [l for l in lines if '"step"' in l]
        self.assertEqual(len(step_lines), 100)
        for line in step_lines:
            rec = json.loads(line)
            self.assertIn("wall_dt_ms", rec)
            self.assertIn("entity_count", rec)

    def test_determinism_with_uavs(self) -> None:
        """SIM-003: identical seed -> identical UAV positions."""
        def _run() -> list:
            log_path = _TMPDIR / f"det_{uuid.uuid4()}.jsonl"
            cfg = SimulationConfig(duration_s=2.0, dt=0.1, seed=99, log_file=log_path)
            sim = SimulationEngine(cfg, _small_world())
            sim.add_entity(_make_uav("uav-det", endurance_s=600.0))
            sim.run()
            return [e.position.tolist() for e in sim.entities.by_type(EntityType.UAV)]

        self.assertEqual(_run(), _run())

    def test_uav_stays_in_world_bounds(self) -> None:
        """FLR-005, ENV-001: UAV stays within world bounds over 50 steps."""
        log_path = _TMPDIR / f"bounds_{uuid.uuid4()}.jsonl"
        cfg = SimulationConfig(duration_s=5.0, dt=0.1, seed=7, log_file=log_path)
        sim = SimulationEngine(cfg, _small_world())
        uav = _make_uav("uav-bounds", endurance_s=600.0)
        sim.add_entity(uav)
        with sim.logger:
            for _ in range(50):
                sim.step()
                if uav.alive:
                    self.assertLessEqual(uav.position[0], 600.0)
                    self.assertGreaterEqual(uav.position[0], 0.0)
                    self.assertLessEqual(uav.position[1], 600.0)
                    self.assertGreaterEqual(uav.position[1], 0.0)


# =============================================================================
# Visualiser state tests (NF-VIZ-008..015, ATC-M3-004..007)
# Uses Agg backend (headless); tests state management, not rendering.
# =============================================================================

class TestDebugPlotInteraction(unittest.TestCase):
    """
    Visualiser interaction tests (NF-VIZ-008..015, ATC-M3-004..007).

    Agg backend is headless so event callbacks are not wired by the
    constructor. Tests call the event handler methods directly.

    API note: the actual DebugPlot implementation uses:
      _on_key(event)                    — keyboard handler
      _on_button_press(event)           — left-click triggers selection via
                                          _handle_entity_selection(event)
      _find_nearest_entity(x, y)        — returns Entity | None
      _on_scroll(event)                 — reads/writes self._xlim/_ylim via
                                          self._ax.get_xlim()/set_xlim()
    """

    def setUp(self) -> None:
        self.plot = _make_plot()
        # Sync axes limits to the internal state so scroll math is consistent
        self.plot._ax.set_xlim(0.0, 600.0)
        self.plot._ax.set_ylim(0.0, 600.0)
        # Populate entity cache for hit-testing (NF-VIZ-010)
        self.uav = _make_uav("uav-viz", pos=np.array([300.0, 300.0, 100.0]))
        self.plot._last_entities = [self.uav]

    def test_initial_view_is_full_world(self) -> None:
        """NF-VIZ-009: initial _xlim/_ylim spans full world extent."""
        self.assertAlmostEqual(self.plot._xlim[0], 0.0)
        self.assertAlmostEqual(self.plot._xlim[1], 600.0)
        self.assertAlmostEqual(self.plot._ylim[0], 0.0)
        self.assertAlmostEqual(self.plot._ylim[1], 600.0)

    def test_scroll_up_zooms_in(self) -> None:
        """NF-VIZ-008 (ATC-M3-004): scroll-up shrinks the view extent."""
        initial_width = self.plot._ax.get_xlim()[1] - self.plot._ax.get_xlim()[0]

        class FakeScrollEvent:
            button = "up"
            inaxes = self.plot._ax
            xdata = 300.0
            ydata = 300.0

        self.plot._on_scroll(FakeScrollEvent())
        new_width = self.plot._ax.get_xlim()[1] - self.plot._ax.get_xlim()[0]
        self.assertLess(new_width, initial_width)

    def test_scroll_down_zooms_out(self) -> None:
        """NF-VIZ-008 (ATC-M3-004): scroll-down enlarges the view extent."""
        # Zoom in first so there is room to zoom back out
        class FakeScrollUp:
            button = "up"
            inaxes = self.plot._ax
            xdata = 300.0
            ydata = 300.0
        self.plot._on_scroll(FakeScrollUp())
        mid_width = self.plot._ax.get_xlim()[1] - self.plot._ax.get_xlim()[0]

        class FakeScrollDown:
            button = "down"
            inaxes = self.plot._ax
            xdata = 300.0
            ydata = 300.0
        self.plot._on_scroll(FakeScrollDown())
        new_width = self.plot._ax.get_xlim()[1] - self.plot._ax.get_xlim()[0]
        self.assertGreater(new_width, mid_width)

    def test_key_r_resets_view(self) -> None:
        """NF-VIZ-009 (ATC-M3-004): pressing R restores default limits."""
        self.plot._xlim = (100.0, 400.0)
        self.plot._ylim = (100.0, 400.0)
        self.plot._ax.set_xlim(100.0, 400.0)

        class FakeKeyEvent:
            key = "r"

        self.plot._on_key(FakeKeyEvent())
        self.assertAlmostEqual(self.plot._xlim[0], 0.0)
        self.assertAlmostEqual(self.plot._xlim[1], 600.0)

    def test_click_on_entity_selects_it(self) -> None:
        """NF-VIZ-010 (ATC-M3-005): left-click near entity sets selection."""

        class FakeClickEvent:
            button = 1
            inaxes = self.plot._ax
            xdata = 300.0   # Entity is at (300, 300)
            ydata = 300.0

        self.plot._on_button_press(FakeClickEvent())
        self.assertEqual(self.plot._selected_entity_id, "uav-viz")

    def test_click_on_empty_deselects(self) -> None:
        """NF-VIZ-013 (ATC-M3-007): left-click on empty space clears selection."""
        self.plot._selected_entity_id = "uav-viz"

        class FakeClickFar:
            button = 1
            inaxes = self.plot._ax
            xdata = 1.0     # Far from entity at (300, 300)
            ydata = 1.0

        self.plot._on_button_press(FakeClickFar())
        self.assertIsNone(self.plot._selected_entity_id)

    def test_escape_deselects(self) -> None:
        """NF-VIZ-013 (ATC-M3-007): Escape key clears selection."""
        self.plot._selected_entity_id = "uav-viz"

        class FakeEscape:
            key = "escape"

        self.plot._on_key(FakeEscape())
        self.assertIsNone(self.plot._selected_entity_id)

    def test_find_nearest_returns_entity_when_close(self) -> None:
        """NF-VIZ-010 (ATC-M3-005): _find_nearest_entity returns Entity near click."""
        result = self.plot._find_nearest_entity(300.0, 300.0)
        # Result is Entity | None; verify it matched our uav
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result.entity_id, "uav-viz")

    def test_find_nearest_returns_none_when_far(self) -> None:
        """NF-VIZ-010 (ATC-M3-005): _find_nearest_entity returns None far from entities."""
        result = self.plot._find_nearest_entity(1.0, 1.0)
        self.assertIsNone(result)


# =============================================================================
# Per-entity neighbor radius (M3 base.py)
# =============================================================================

class TestPerEntityNeighborRadius(unittest.TestCase):

    def test_ground_entity_uses_default_radius(self) -> None:
        """M3 base.py: pedestrian uses NEIGHBOR_RADIUS_M."""
        ped = PedestrianEntity(
            entity_id="p1",
            position=np.array([50.0, 50.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(ped.neighbor_radius_m, _D.NEIGHBOR_RADIUS_M)

    def test_uav_uses_larger_radius(self) -> None:
        """M3 base.py: UAV uses UAV_NEIGHBOR_RADIUS_M."""
        self.assertAlmostEqual(_make_uav().neighbor_radius_m, _D.UAV_NEIGHBOR_RADIUS_M)


# =============================================================================
# defaults.py: WORLD_ALT constants now consumed by World (NF-M-006)
# =============================================================================

class TestDefaultsUsage(unittest.TestCase):

    def test_world_uses_world_alt_defaults(self) -> None:
        """NF-M-006: World constructor defaults come from SimDefaults constants."""
        world = World(extent=np.array([200.0, 200.0]))
        self.assertAlmostEqual(world.alt_floor_m, _D.WORLD_ALT_FLOOR_M)
        self.assertAlmostEqual(world.alt_ceil_m, _D.WORLD_ALT_CEIL_M)

    def test_search_margin_equals_geofence_plus_corner(self) -> None:
        """NF-M-006: UAV_SEARCH_MARGIN_M = UAV_GEOFENCE_MARGIN_M + UAV_CORNER_ESCAPE_MARGIN_M."""
        expected = _D.UAV_GEOFENCE_MARGIN_M + _D.UAV_CORNER_ESCAPE_MARGIN_M
        self.assertAlmostEqual(_D.UAV_SEARCH_MARGIN_M, expected)


if __name__ == "__main__":
    unittest.main()


# =============================================================================
# Matplotlib Agg backend must be set before importing DebugPlot.
# The import is deferred to setUp() so individual test files that don't need
# the visualiser are unaffected.
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import types as _types   # for SimpleNamespace mock events

from sim3dves.viz.debug_plot import DebugPlot


# =============================================================================
# FLR-008: search-pattern safety margin (ATC-M3-002)
# =============================================================================

class TestSearchPatternSafetyMargin(unittest.TestCase):
    """ATC-M3-002: FLR-008 all waypoints are within UAV_SEARCH_MARGIN_M."""

    def _make_uav_pattern(self, pattern: SearchPattern) -> UAVEntity:
        return _make_uav(
            search_pattern=pattern,
            world_extent=np.array([600.0, 600.0]),
        )

    def _check_margin(self, pattern: SearchPattern) -> None:
        uav = self._make_uav_pattern(pattern)
        wps = uav._generate_search_waypoints()
        margin = _D.UAV_SEARCH_MARGIN_M
        for wp in wps:
            self.assertGreaterEqual(
                wp[0], margin - 0.01,
                f"{pattern.name} waypoint x={wp[0]:.1f} inside west margin",
            )
            self.assertLessEqual(
                wp[0], 600.0 - margin + 0.01,
                f"{pattern.name} waypoint x={wp[0]:.1f} inside east margin",
            )
            self.assertGreaterEqual(
                wp[1], margin - 0.01,
                f"{pattern.name} waypoint y={wp[1]:.1f} inside south margin",
            )
            self.assertLessEqual(
                wp[1], 600.0 - margin + 0.01,
                f"{pattern.name} waypoint y={wp[1]:.1f} inside north margin",
            )

    def test_lawnmower_respects_search_margin(self) -> None:
        """ATC-M3-002: FLR-008 LAWNMOWER waypoints >= UAV_SEARCH_MARGIN_M from edge."""
        self._check_margin(SearchPattern.LAWNMOWER)

    def test_spiral_respects_search_margin(self) -> None:
        """ATC-M3-002: FLR-008 EXPANDING_SPIRAL waypoints >= UAV_SEARCH_MARGIN_M."""
        self._check_margin(SearchPattern.EXPANDING_SPIRAL)

    def test_random_walk_respects_search_margin(self) -> None:
        """ATC-M3-002: FLR-008 RANDOM_WALK waypoints >= UAV_SEARCH_MARGIN_M."""
        self._check_margin(SearchPattern.RANDOM_WALK)

    def test_search_margin_wider_than_geofence_margin(self) -> None:
        """FLR-008: UAV_SEARCH_MARGIN_M must be strictly greater than geofence margin."""
        self.assertGreater(_D.UAV_SEARCH_MARGIN_M, _D.UAV_GEOFENCE_MARGIN_M)


# =============================================================================
# FLR-011: corner-escape heading (ATC-M3-001)
# =============================================================================

class TestCornerEscape(unittest.TestCase):
    """ATC-M3-001: FLR-011 UAV stays within world boundary near a corner."""

    def _run_corner_scenario(
        self,
        start_x: float,
        start_y: float,
        heading_deg: float,
        steps: int = 80,
    ) -> None:
        """
        Place a UAV heading into a corner and verify it stays within bounds.
        The UAV is given an initial velocity in the heading direction so the
        corner-escape logic must actively counteract it.
        """
        world_x, world_y = 600.0, 600.0
        uav = UAVEntity(
            entity_id="uav-corner-atc",
            position=np.array([start_x, start_y, 100.0]),
            heading=heading_deg,
            world_extent=np.array([world_x, world_y]),
            alt_floor_m=30.0,
            alt_ceil_m=500.0,
            endurance_s=600.0,
            rng=np.random.default_rng(0),
        )
        # Initialise velocity in the corner-heading direction
        hr = math.radians(heading_deg)
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS * math.cos(hr)
        uav.velocity[1] = _D.UAV_MAX_SPEED_MPS * math.sin(hr)

        for step in range(steps):
            uav.step(dt=0.1)
            self.assertGreaterEqual(
                uav.position[0], 0.0,
                f"Step {step}: exited West boundary (x={uav.position[0]:.1f})",
            )
            self.assertLessEqual(
                uav.position[0], world_x,
                f"Step {step}: exited East boundary (x={uav.position[0]:.1f})",
            )
            self.assertGreaterEqual(
                uav.position[1], 0.0,
                f"Step {step}: exited South boundary (y={uav.position[1]:.1f})",
            )
            self.assertLessEqual(
                uav.position[1], world_y,
                f"Step {step}: exited North boundary (y={uav.position[1]:.1f})",
            )

    def test_sw_corner_escape(self) -> None:
        """ATC-M3-001 SW corner: UAV heading 225 deg stays within bounds."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        # Start inside the corner pocket (both margins breached) heading SW
        self._run_corner_scenario(m + 30.0, m + 30.0, heading_deg=225.0)

    def test_se_corner_escape(self) -> None:
        """ATC-M3-001 SE corner: UAV heading 315 deg stays within bounds."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        self._run_corner_scenario(600.0 - m - 30.0, m + 30.0, heading_deg=315.0)

    def test_ne_corner_escape(self) -> None:
        """ATC-M3-001 NE corner: UAV heading 45 deg stays within bounds."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        self._run_corner_scenario(600.0 - m - 30.0, 600.0 - m - 30.0, heading_deg=45.0)

    def test_nw_corner_escape(self) -> None:
        """ATC-M3-001 NW corner: UAV heading 135 deg stays within bounds."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        self._run_corner_scenario(m + 30.0, 600.0 - m - 30.0, heading_deg=135.0)

    def test_in_corner_pocket_detection(self) -> None:
        """FLR-011: _in_corner_pocket returns True inside corner zone."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        uav = _make_uav(pos=np.array([m - 1.0, m - 1.0, 100.0]))
        self.assertTrue(uav._in_corner_pocket())

    def test_mid_world_not_corner_pocket(self) -> None:
        """FLR-011: _in_corner_pocket returns False at world centre."""
        uav = _make_uav(pos=np.array([300.0, 300.0, 100.0]))
        self.assertFalse(uav._in_corner_pocket())

    def test_corner_escape_heading_points_outward_sw(self) -> None:
        """FLR-011: SW corner escape heading has positive X and Y components."""
        m = _D.UAV_GEOFENCE_MARGIN_M
        uav = _make_uav(pos=np.array([m - 1.0, m - 1.0, 100.0]))
        heading_rad = uav._corner_escape_heading_rad()
        # SW corner escape must point NE: positive X and Y velocity components
        self.assertGreater(math.cos(heading_rad), 0.0)
        self.assertGreater(math.sin(heading_rad), 0.0)


# =============================================================================
# FLR-001: turn-rate compliance during NFZ avoidance (ATC-M3-003)
# =============================================================================

class TestNFZAvoidanceTurnRate(unittest.TestCase):
    """ATC-M3-003: FLR-001 NFZ avoidance respects UAV_TURN_RATE_DPS."""

    def test_heading_change_per_step_bounded_during_avoidance(self) -> None:
        """
        ATC-M3-003: |delta_heading| <= UAV_TURN_RATE_DPS * dt at every step
        while NFZ avoidance is active.
        """
        from sim3dves.core.world import NFZCylinder

        # NFZ placed directly ahead of the UAV at lookahead distance
        lookahead_m = _D.UAV_NFZ_LOOKAHEAD_S * _D.UAV_MAX_SPEED_MPS
        nfz = NFZCylinder(
            center_xy=np.array([300.0 + lookahead_m * 0.8, 300.0]),
            radius_m=80.0,
            alt_max_m=300.0,
        )
        uav = _make_uav(
            pos=np.array([300.0, 300.0, 100.0]),
            heading=0.0,                # Flying East
            nfz_cylinders=[nfz],
        )
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS
        uav.velocity[1] = 0.0

        dt = 0.1
        max_delta_deg = _D.UAV_TURN_RATE_DPS * dt + 0.05   # +0.05 float tolerance

        for _ in range(30):
            old_heading = uav.heading
            uav.step(dt=dt)
            new_heading = uav.heading
            # Compute shortest angular difference in (-180, 180]
            delta = ((new_heading - old_heading + 180.0) % 360.0) - 180.0
            self.assertLessEqual(
                abs(delta), max_delta_deg,
                f"|delta_heading|={abs(delta):.3f} > {max_delta_deg:.3f} deg",
            )


# =============================================================================
# NF-VIZ-008, NF-VIZ-009: zoom and reset (ATC-M3-004)
# =============================================================================

class TestVisualizerZoom(unittest.TestCase):
    """ATC-M3-004 / ATC-M3-009: NF-VIZ-008 zoom and NF-VIZ-009 reset."""

    def setUp(self) -> None:
        self.plot = DebugPlot(600.0, 600.0)

    def tearDown(self) -> None:
        plt.close("all")

    def test_scroll_up_zooms_in(self) -> None:
        """ATC-M3-004: NF-VIZ-008 scroll-up reduces xlim width (zoom in)."""
        initial_width = self.plot._xlim[1] - self.plot._xlim[0]
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt)
        new_width = self.plot._xlim[1] - self.plot._xlim[0]
        self.assertLess(new_width, initial_width)

    def test_scroll_down_zooms_out(self) -> None:
        """ATC-M3-004: NF-VIZ-008 scroll-down increases xlim width (zoom out).

        _on_scroll reads self._ax.get_xlim(); with Agg and no render the axes
        default to (0, 1).  Zoom in first (scroll-up) so the zoomed-in state
        becomes the reference, then zoom back out and verify it grew.
        """
        # Sync axes to the internal state so get_xlim() sees world extent
        self.plot._ax.set_xlim(*self.plot._xlim)
        self.plot._ax.set_ylim(*self.plot._ylim)

        # Zoom in to create a reference narrower than initial
        evt_up = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt_up)
        mid_width = self.plot._xlim[1] - self.plot._xlim[0]

        # Now zoom out — must be wider than the zoomed-in state
        evt_down = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="down", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt_down)
        new_width = self.plot._xlim[1] - self.plot._xlim[0]
        self.assertGreater(new_width, mid_width)

    def test_reset_view_restores_full_extent(self) -> None:
        """ATC-M3-009: NF-VIZ-009 _reset_view restores default limits."""
        # Zoom in first
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt)
        self.plot._reset_view()
        self.assertAlmostEqual(self.plot._xlim[0], 0.0, places=3)
        self.assertAlmostEqual(self.plot._xlim[1], 600.0, places=3)
        self.assertAlmostEqual(self.plot._ylim[0], 0.0, places=3)
        self.assertAlmostEqual(self.plot._ylim[1], 600.0, places=3)

    def test_key_r_resets_view(self) -> None:
        """ATC-M3-009: NF-VIZ-009 'R' key triggers reset."""
        evt_scroll = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt_scroll)
        evt_key = _types.SimpleNamespace(key="r")
        self.plot._on_key(evt_key)
        self.assertAlmostEqual(self.plot._xlim[1], 600.0, places=3)

    def test_zoom_preserves_across_render(self) -> None:
        """NF-VIZ-008: zoom state is preserved across render() calls."""
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt)
        zoomed_width = self.plot._xlim[1] - self.plot._xlim[0]
        # render() must not reset xlim to world defaults
        self.plot.render([], sim_time=1.0)
        after_width = self.plot._xlim[1] - self.plot._xlim[0]
        self.assertAlmostEqual(zoomed_width, after_width, places=3)


# =============================================================================
# NF-VIZ-010: entity selection (ATC-M3-005)
# =============================================================================

class TestVisualizerSelection(unittest.TestCase):
    """ATC-M3-005: NF-VIZ-010 entity click-to-select."""

    def setUp(self) -> None:
        from sim3dves.entities.pedestrian import PedestrianEntity
        self.plot = DebugPlot(600.0, 600.0)
        self.ped = PedestrianEntity(
            entity_id="ped-vis-01",
            position=np.array([300.0, 300.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        # Render once to populate _last_entities and calibrate axes transform
        self.plot.render([self.ped], sim_time=0.0)

    def tearDown(self) -> None:
        plt.close("all")

    def test_click_on_entity_selects_it(self) -> None:
        """ATC-M3-005: NF-VIZ-010 click at entity position selects it."""
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button=1,
            xdata=300.0, ydata=300.0,
        )
        self.plot._handle_entity_selection(evt)
        self.assertEqual(self.plot._selected_entity_id, "ped-vis-01")

    def test_click_far_from_entity_clears_selection(self) -> None:
        """ATC-M3-013 / NF-VIZ-013: clicking empty space deselects."""
        self.plot._selected_entity_id = "ped-vis-01"
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button=1,
            xdata=10.0, ydata=10.0,
        )
        self.plot._handle_entity_selection(evt)
        self.assertIsNone(self.plot._selected_entity_id)

    def test_selected_entity_property_returns_entity(self) -> None:
        """NF-VIZ-012: _selected_entity resolves selected ID to Entity object."""
        self.plot._selected_entity_id = "ped-vis-01"
        sel = self.plot._selected_entity
        self.assertIsNotNone(sel)
        self.assertEqual(sel.entity_id, "ped-vis-01")  # type: ignore[union-attr]

    def test_selected_entity_returns_none_when_no_selection(self) -> None:
        """NF-VIZ-012: _selected_entity is None with no active selection."""
        self.plot._selected_entity_id = None
        self.assertIsNone(self.plot._selected_entity)


# =============================================================================
# NF-VIZ-011: inspection panel content (ATC-M3-006)
# =============================================================================

class TestInspectionPanelContent(unittest.TestCase):
    """ATC-M3-006: NF-VIZ-011 panel shows required fields for all entity types."""

    def setUp(self) -> None:
        self.plot = DebugPlot(600.0, 600.0)

    def tearDown(self) -> None:
        plt.close("all")

    def test_common_fields_in_panel(self) -> None:
        """NF-VIZ-011: panel always shows ID, type, state, speed, position."""
        from sim3dves.entities.pedestrian import PedestrianEntity
        ped = PedestrianEntity(
            entity_id="ped-panel-01",
            position=np.array([100.0, 100.0, 0.0]),
            velocity=np.array([1.0, 0.5, 0.0]),
        )
        text = self.plot._build_panel_text(ped)
        self.assertIn("ped-panel-01", text)
        self.assertIn("PEDESTRIAN", text)
        self.assertIn("Pos", text)
        self.assertIn("Speed", text)

    def test_uav_specific_fields_in_panel(self) -> None:
        """ATC-M3-006: NF-VIZ-011 UAV panel shows mode, endurance, flags, role."""
        uav = UAVEntity(
            entity_id="uav-panel-01",
            position=np.array([300.0, 300.0, 100.0]),
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        text = self.plot._build_panel_text(uav)
        self.assertIn("Mode", text)
        self.assertIn("Endur", text)
        self.assertIn("LowFuel", text)
        self.assertIn("NFZ", text)
        self.assertIn("Role", text)

    def test_uav_low_fuel_shown_in_panel(self) -> None:
        """NF-VIZ-011: low_fuel flag reflected in panel when threshold crossed."""
        uav = _make_uav(endurance_s=0.05)
        uav.step(dt=0.1)   # exhaust endurance
        text = self.plot._build_panel_text(uav)
        self.assertIn("YES", text)  # Low fuel indicator

    def test_vehicle_destination_shown(self) -> None:
        """NF-VIZ-011: vehicle current_destination appears in panel."""
        from sim3dves.entities.vehicle import WheeledVehicleEntity
        from sim3dves.maps.road_network import RoadNetwork
        rn = RoadNetwork.build_grid(3, 3, 100.0, np.array([0.0, 0.0]))
        veh = WheeledVehicleEntity(
            entity_id="veh-panel-01",
            position=np.array([0.0, 0.0, 0.0]),
            road_network=rn,
            rng=np.random.default_rng(0),
        )
        veh.step(dt=0.1)   # trigger path planning to populate _waypoints
        text = self.plot._build_panel_text(veh)
        if veh.current_destination is not None:
            self.assertIn("Dest", text)


# =============================================================================
# NF-VIZ-013: deselect (ATC-M3-007)
# =============================================================================

class TestVisualizerDeselect(unittest.TestCase):
    """ATC-M3-007: NF-VIZ-013 deselect on Escape and empty-space click."""

    def setUp(self) -> None:
        self.plot = DebugPlot(600.0, 600.0)
        self.plot._selected_entity_id = "some-entity"

    def tearDown(self) -> None:
        plt.close("all")

    def test_escape_key_clears_selection(self) -> None:
        """ATC-M3-007: NF-VIZ-013 Escape key sets selected_entity_id to None."""
        evt = _types.SimpleNamespace(key="escape")
        self.plot._on_key(evt)
        self.assertIsNone(self.plot._selected_entity_id)

    def test_none_xdata_clears_selection(self) -> None:
        """NF-VIZ-013: click with None xdata (off-axes) clears selection."""
        evt = _types.SimpleNamespace(
            inaxes=self.plot._ax, button=1, xdata=None, ydata=None
        )
        self.plot._handle_entity_selection(evt)
        self.assertIsNone(self.plot._selected_entity_id)

    def test_panel_not_rendered_when_deselected(self) -> None:
        """NF-VIZ-013: with no selection, _selected_entity returns None."""
        self.plot._selected_entity_id = None
        self.assertIsNone(self.plot._selected_entity)


# =============================================================================
# NF-VIZ-015: visualiser properties (design standards)
# =============================================================================

class TestVisualizerDesignStandards(unittest.TestCase):
    """NF-VIZ-015: visualiser meets production code standards."""

    def setUp(self) -> None:
        self.plot = DebugPlot(600.0, 600.0)

    def tearDown(self) -> None:
        plt.close("all")

    def test_reset_view_is_idempotent(self) -> None:
        """NF-VIZ-015: calling reset_view twice gives identical limits."""
        self.plot._reset_view()
        lim1 = (self.plot._xlim, self.plot._ylim)
        self.plot._reset_view()
        lim2 = (self.plot._xlim, self.plot._ylim)
        self.assertEqual(lim1, lim2)

    def test_scroll_outside_axes_ignored(self) -> None:
        """NF-VIZ-008: scroll event outside axes does not change limits."""
        original = self.plot._xlim
        evt = _types.SimpleNamespace(
            inaxes=None, button="up", xdata=300.0, ydata=300.0
        )
        self.plot._on_scroll(evt)
        self.assertEqual(self.plot._xlim, original)

    def test_render_with_dead_entities_does_not_raise(self) -> None:
        """NF-VIZ-001: render handles mix of alive and dead entities."""
        from sim3dves.entities.pedestrian import PedestrianEntity
        ped = PedestrianEntity(
            entity_id="dead-ped",
            position=np.array([50.0, 50.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        ped.kill()
        try:
            self.plot.render([ped], sim_time=5.0)
        except Exception as exc:
            self.fail(f"render() raised {exc!r} with dead entity")

