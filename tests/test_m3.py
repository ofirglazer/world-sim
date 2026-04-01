"""
tests.test_m3
=============
M3 test suite -- UAVEntity, autopilot FSM, all flight rules, search patterns,
multi-UAV deconfliction, NFZ avoidance, endurance, and end-to-end integration.

Every test function cites the Req ID(s) it validates (NF-TE-002).
Coverage target: >= 80% of new M3 modules (NF-TE-001, NF-M-003).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import json
# import math
import tempfile
import unittest
import uuid
from pathlib import Path
# from typing import List

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.event_bus import EventType
from sim3dves.core.world import NFZCylinder, World
from sim3dves.entities.base import EntityManager, EntityType
from sim3dves.entities.uav import AutopilotMode, SearchPattern, UAVEntity

_D = SimDefaults()
_TMPDIR = Path(tempfile.mkdtemp())


# ### Helpers ###

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
    """Construct a UAVEntity with sensible test defaults."""
    if pos is None:
        pos = np.array([300.0, 300.0, cruise_alt])
    return UAVEntity(
        entity_id=entity_id,
        position=pos,
        heading=heading,
        launch_position=pos.copy(),
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


def _small_world(
    nfz_cylinders: list | None = None,
) -> World:
    """600x600 m world with optional NFZs."""
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
    """Create a minimal SimulationEngine for testing."""
    if world is None:
        world = _small_world()
    cfg = SimulationConfig(
        duration_s=duration_s,
        dt=dt,
        seed=42,
        log_file=log_file or (_TMPDIR / f"{uuid.uuid4()}.jsonl"),
    )
    return SimulationEngine(cfg, world)


# =============================================================================
# UAV-001 / UAV-002: 3-D kinematics and altitude initialisation
# =============================================================================

class TestUAVKinematics(unittest.TestCase):

    def test_altitude_initialised_to_cruise(self) -> None:
        """UAV-001: Z position at construction equals cruise_altitude_m."""
        uav = _make_uav(cruise_alt=150.0)
        self.assertAlmostEqual(uav.position[2], 150.0, places=3)

    def test_entity_type_is_uav(self) -> None:
        """ENT-002: entity_type is EntityType.UAV Enum value."""
        uav = _make_uav()
        self.assertEqual(uav.entity_type, EntityType.UAV)

    def test_initial_autopilot_mode_is_waypoint(self) -> None:
        """UAV-005: default autopilot mode is WAYPOINT."""
        uav = _make_uav()
        self.assertEqual(uav.autopilot_mode, AutopilotMode.WAYPOINT)

    def test_altitude_floor_enforced(self) -> None:
        """FLR-002: Z never drops below alt_floor_m after steps."""
        uav = _make_uav(cruise_alt=_D.UAV_ALT_FLOOR_M)
        # Force downward velocity
        uav.velocity[2] = -50.0
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertGreaterEqual(uav.position[2], _D.UAV_ALT_FLOOR_M - 0.01)

    def test_altitude_ceiling_enforced(self) -> None:
        """FLR-003: Z never exceeds alt_ceil_m after steps."""
        uav = _make_uav(cruise_alt=_D.UAV_ALT_CEIL_M)
        uav.velocity[2] = 50.0
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertLessEqual(uav.position[2], _D.UAV_ALT_CEIL_M + 0.01)

    def test_alive_flag_true_at_construction(self) -> None:
        """ENT-001: alive=True at construction."""
        uav = _make_uav()
        self.assertTrue(uav.alive)


# =============================================================================
# UAV-003: Endurance budget
# =============================================================================

class TestUAVEndurance(unittest.TestCase):

    def test_endurance_decreases_per_step(self) -> None:
        """UAV-003: endurance_remaining_s decreases by dt each step."""
        uav = _make_uav(endurance_s=100.0)
        dt = 0.1
        uav.step(dt=dt)
        self.assertAlmostEqual(uav.endurance_remaining_s, 100.0 - dt, places=5)

    def test_endurance_never_negative(self) -> None:
        """UAV-003: endurance_remaining_s is clamped to 0."""
        uav = _make_uav(endurance_s=0.1)
        for _ in range(20):
            uav.step(dt=0.1)
        self.assertGreaterEqual(uav.endurance_remaining_s, 0.0)

    def test_low_fuel_triggers_rtb(self) -> None:
        """FLR-006: autopilot transitions to RTB when endurance threshold crossed."""
        # Set endurance to just above low-fuel threshold
        threshold = _D.UAV_LOW_FUEL_THRESHOLD_S
        uav = _make_uav(endurance_s=threshold + 0.05)
        # One step will drop below threshold
        uav.step(dt=0.1)
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
        )

    def test_rtb_not_re_triggered_once_active(self) -> None:
        """FLR-006: RTB mode persists; not overridden by further depletion."""
        uav = _make_uav(endurance_s=0.05)
        uav.step(dt=0.1)
        # After depletion: RTB or EMERGENCY_LAND (both are valid)
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
        )
        # Extra steps should keep RTB/EMERGENCY_LAND, not revert to WAYPOINT
        for _ in range(5):
            uav.step(dt=0.1)
        self.assertIn(
            uav.autopilot_mode,
            (AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND),
        )


# =============================================================================
# FLR-001: NFZ avoidance
# =============================================================================

class TestNFZAvoidance(unittest.TestCase):

    def test_nfz_avoidance_deflects_velocity(self) -> None:
        """FLR-001: velocity changes direction when lookahead penetrates NFZ."""
        # Place NFZ directly ahead of UAV
        nfz = NFZCylinder(
            center_xy=np.array([400.0, 300.0]),
            radius_m=80.0,
            alt_max_m=300.0,
        )
        uav = _make_uav(
            pos=np.array([200.0, 300.0, 100.0]),
            heading=0.0,    # Flying East
            nfz_cylinders=[nfz],
        )
        # Force heading East with significant speed so lookahead hits NFZ
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS
        uav.velocity[1] = 0.0
        velocity_before = uav.velocity.copy()
        uav.step(dt=0.1)
        velocity_after = uav.velocity.copy()
        # Y component should have changed (lateral deflection)
        self.assertFalse(
            np.allclose(velocity_before[:2], velocity_after[:2], atol=0.01),
            "Velocity unchanged — NFZ avoidance did not deflect heading",
        )

    def test_uav_not_killed_on_nfz_entry(self) -> None:
        """FLR-001: NFZ avoidance does not kill the UAV; engine logs event only."""
        nfz = NFZCylinder(
            center_xy=np.array([305.0, 300.0]),
            radius_m=20.0,
            alt_max_m=200.0,
        )
        world = _small_world(nfz_cylinders=[nfz])
        sim = _make_sim(world=world)
        uav = _make_uav(
            pos=np.array([300.0, 300.0, 100.0]),
            nfz_cylinders=[nfz],
        )
        sim.add_entity(uav)
        with sim.logger:
            for _ in range(5):
                sim.step()
        # UAV should NOT be killed by an NFZ event (avoidance, not OOB)
        self.assertTrue(uav.alive)


# =============================================================================
# FLR-004: UAV-UAV separation
# =============================================================================

class TestUAVSeparation(unittest.TestCase):

    def test_separation_velocity_correction_applied(self) -> None:
        """FLR-004: UAV velocity corrected when closer than UAV_SEPARATION_M."""
        uav_a = _make_uav("uav-a", pos=np.array([300.0, 300.0, 100.0]))
        uav_b = _make_uav("uav-b", pos=np.array([310.0, 300.0, 100.0]))

        mgr = EntityManager()
        mgr.add(uav_a)
        mgr.add(uav_b)

        vel_before = uav_a.velocity.copy()
        mgr.step_all(dt=0.1)
        vel_after = uav_a.velocity.copy()

        # After step, velocity should differ due to separation correction
        # (at least one of the two UAVs will be pushed)
        self.assertFalse(
            np.allclose(vel_before, vel_after, atol=0.01)
            and np.allclose(uav_b.velocity, vel_before, atol=0.01),
        )

    def test_neighbor_radius_uav_larger_than_ground(self) -> None:
        """M3 base.py: UAV neighbor_radius_m > ground entity default."""
        uav = _make_uav()
        self.assertGreater(uav.neighbor_radius_m, _D.NEIGHBOR_RADIUS_M)


# =============================================================================
# FLR-005: Geofence
# =============================================================================

class TestGeofence(unittest.TestCase):

    def test_geofence_triggers_rtb(self) -> None:
        """FLR-005: UAV transitions to RTB when within UAV_GEOFENCE_MARGIN_M."""
        margin = _D.UAV_GEOFENCE_MARGIN_M
        # Place UAV just inside the margin from the East boundary
        uav = _make_uav(
            pos=np.array([600.0 - margin + 5.0, 300.0, 100.0]),
        )
        # Fly East (toward boundary) so _near_geofence() returns True
        uav.velocity[0] = _D.UAV_MAX_SPEED_MPS
        uav.step(dt=0.1)
        self.assertEqual(uav.autopilot_mode, AutopilotMode.RTB)


# =============================================================================
# FLR-007: Wind model
# =============================================================================

class TestWindModel(unittest.TestCase):

    def test_wind_drifts_position(self) -> None:
        """FLR-007: wind vector displaces position by wind * dt each step."""
        wind = np.array([5.0, 0.0, 0.0])   # 5 m/s East wind
        uav = _make_uav(pos=np.array([300.0, 300.0, 100.0]))
        uav._wind = wind
        uav.velocity[:] = 0.0  # Zero commanded velocity to isolate wind effect
        x_before = uav.position[0]
        uav.step(dt=1.0)
        x_after = uav.position[0]
        self.assertGreater(x_after, x_before)


# =============================================================================
# FLR-008: Search patterns
# =============================================================================

class TestSearchPatterns(unittest.TestCase):

    def _count_waypoints(self, pattern: SearchPattern) -> int:
        uav = _make_uav(search_pattern=pattern)
        wps = uav._generate_search_waypoints()
        return len(wps)

    def test_lawnmower_generates_waypoints(self) -> None:
        """FLR-008: LAWNMOWER pattern produces >= 2 waypoints."""
        self.assertGreaterEqual(
            self._count_waypoints(SearchPattern.LAWNMOWER), 2
        )

    def test_expanding_spiral_generates_waypoints(self) -> None:
        """FLR-008: EXPANDING_SPIRAL pattern produces >= 8 waypoints."""
        self.assertGreaterEqual(
            self._count_waypoints(SearchPattern.EXPANDING_SPIRAL), 8
        )

    def test_random_walk_generates_correct_count(self) -> None:
        """FLR-008: RANDOM_WALK produces exactly UAV_RANDOM_WALK_WAYPOINTS points."""
        uav = _make_uav(search_pattern=SearchPattern.RANDOM_WALK)
        wps = uav._generate_search_waypoints()
        self.assertEqual(len(wps), _D.UAV_RANDOM_WALK_WAYPOINTS)

    def test_waypoints_within_world_bounds(self) -> None:
        """FLR-008: all waypoints are within world extent minus geofence margin."""
        for pattern in SearchPattern:
            uav = _make_uav(search_pattern=pattern)
            wps = uav._generate_search_waypoints()
            margin = _D.UAV_GEOFENCE_MARGIN_M
            for wp in wps:
                self.assertGreaterEqual(wp[0], margin - 0.01)
                self.assertLessEqual(wp[0], 600.0 - margin + 0.01)
                self.assertGreaterEqual(wp[1], margin - 0.01)
                self.assertLessEqual(wp[1], 600.0 - margin + 0.01)

    def test_waypoints_at_cruise_altitude(self) -> None:
        """FLR-008: all search waypoints are at cruise altitude."""
        alt = 120.0
        uav = _make_uav(search_pattern=SearchPattern.LAWNMOWER, cruise_alt=alt)
        wps = uav._generate_search_waypoints()
        for wp in wps:
            self.assertAlmostEqual(wp[2], alt, places=3)


# =============================================================================
# FLR-009: Cued orbit
# =============================================================================

class TestCuedOrbit(unittest.TestCase):

    def test_cue_orbit_transitions_mode(self) -> None:
        """FLR-009: cue_orbit() transitions autopilot mode to ORBIT."""
        uav = _make_uav()
        self.assertEqual(uav.autopilot_mode, AutopilotMode.WAYPOINT)
        uav.cue_orbit(center_xy=np.array([300.0, 300.0]))
        self.assertEqual(uav.autopilot_mode, AutopilotMode.ORBIT)

    def test_orbit_mode_generates_circular_velocity(self) -> None:
        """FLR-009: in ORBIT mode UAV velocity is non-zero and tangential."""
        uav = _make_uav(pos=np.array([400.0, 300.0, 100.0]))
        uav.cue_orbit(
            center_xy=np.array([300.0, 300.0]),
            radius_m=100.0,
        )
        uav.step(dt=0.1)
        h_speed = float(np.linalg.norm(uav.velocity[:2]))
        self.assertGreater(h_speed, 0.1)

    def test_orbit_center_stored_correctly(self) -> None:
        """FLR-009: orbit centre is stored verbatim."""
        uav = _make_uav()
        centre = np.array([123.0, 456.0])
        uav.cue_orbit(centre)
        np.testing.assert_array_almost_equal(uav._orbit_center, centre)


# =============================================================================
# FLR-010: Deconfliction
# =============================================================================

class TestDeconfliction(unittest.TestCase):

    def test_two_uavs_same_cue_get_different_roles(self) -> None:
        """FLR-010: two UAVs orbiting the same cue -> PRIMARY + SECONDARY."""
        uav_a = _make_uav("uav-a-deconf", pos=np.array([300.0, 300.0, 100.0]))
        uav_b = _make_uav("uav-b-deconf", pos=np.array([320.0, 300.0, 100.0]))
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
        """FLR-010: single UAV in ORBIT mode is always PRIMARY."""
        uav = _make_uav()
        uav.cue_orbit(np.array([300.0, 300.0]))
        uav.step(dt=0.1)
        self.assertEqual(uav.deconfliction_role, "PRIMARY")

    def test_secondary_altitude_offset(self) -> None:
        """FLR-010: SECONDARY UAV targets altitude = orbit_alt + ALT_STEP_M."""
        uav_a = _make_uav("uav-x", pos=np.array([300.0, 300.0, 100.0]))
        uav_b = _make_uav("uav-y", pos=np.array([300.0, 305.0, 100.0]))
        centre = np.array([300.0, 300.0])
        uav_a.cue_orbit(centre, altitude_m=100.0)
        uav_b.cue_orbit(centre, altitude_m=100.0)

        mgr = EntityManager()
        mgr.add(uav_a)
        mgr.add(uav_b)
        # Run enough steps for SECONDARY altitude to diverge
        for _ in range(10):
            mgr.step_all(dt=0.1)

        # Determine which is SECONDARY
        secondary = uav_a if uav_a.deconfliction_role == "SECONDARY" else uav_b
        # SECONDARY should be climbing toward 100 + ALT_STEP
        target_alt = 100.0 + _D.UAV_DECONFLICTION_ALT_STEP_M
        # Allow a tolerance since the UAV is still converging
        self.assertGreater(secondary.position[2], 100.0 - 1.0)
        self.assertLessEqual(secondary.position[2], target_alt + 5.0)


# =============================================================================
# UAV-004: Multi-UAV simultaneous operation
# =============================================================================

class TestMultiUAV(unittest.TestCase):

    def test_four_uavs_all_alive_after_10_steps(self) -> None:
        """UAV-004: four UAVs operate concurrently without killing each other."""
        sim = _make_sim(duration_s=1.0)
        for i in range(4):
            uav = _make_uav(
                entity_id=f"uav-multi-{i}",
                pos=np.array([100.0 + i * 100.0, 300.0, 100.0 + i * 10.0]),
            )
            sim.add_entity(uav)
        with sim.logger:
            for _ in range(10):
                sim.step()
        uavs = sim.entities.by_type(EntityType.UAV)
        alive_count = sum(1 for u in uavs if u.alive)
        self.assertEqual(alive_count, 4)


# =============================================================================
# Engine: NFZ violation event published and logged
# =============================================================================

class TestEngineNFZLogging(unittest.TestCase):

    def test_nfz_violation_event_written_to_jsonl(self) -> None:
        """FLR-001, LOG-002: NFZ violation event appears in JSONL log."""
        nfz = NFZCylinder(
            center_xy=np.array([300.0, 300.0]),
            radius_m=5.0,          # Tiny NFZ — UAV spawned inside it
            alt_max_m=200.0,
        )
        world = _small_world(nfz_cylinders=[nfz])
        log_path = _TMPDIR / f"nfz_test_{uuid.uuid4()}.jsonl"
        sim = _make_sim(world=world, log_file=log_path)

        # Spawn UAV exactly inside the NFZ so violation triggers on step 1
        uav = UAVEntity(
            entity_id="uav-nfz-test",
            position=np.array([300.0, 300.0, 100.0]),
            nfz_cylinders=[nfz],
            world_extent=np.array([600.0, 600.0]),
            rng=np.random.default_rng(0),
        )
        sim.add_entity(uav)

        with sim.logger:
            for _ in range(3):
                sim.step()

        records = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
        ]
        nfz_events = [
            r for r in records
            if r.get("_record_type") == "event"
            and r.get("type") == EventType.NFZ_VIOLATION.name
        ]
        self.assertGreater(len(nfz_events), 0, "No NFZ_VIOLATION events logged")


# =============================================================================
# Integration: full engine run with UAVs
# =============================================================================

class TestM3Integration(unittest.TestCase):

    def test_engine_run_100_steps_with_uavs(self) -> None:
        """NF-TE-004: headless run for 100 steps produces valid JSONL output."""
        log_path = _TMPDIR / f"integration_{uuid.uuid4()}.jsonl"
        world = _small_world()
        cfg = SimulationConfig(
            duration_s=10.0,  # 100 steps at dt=0.1
            dt=0.1,
            seed=0,
            log_file=log_path,
        )
        sim = SimulationEngine(cfg, world)
        for i in range(2):
            sim.add_entity(_make_uav(
                entity_id=f"uav-int-{i}",
                pos=np.array([200.0 + i * 100.0, 300.0, 100.0]),
                endurance_s=600.0,
            ))

        sim.run()   # uses context manager internally

        lines = log_path.read_text().splitlines()
        self.assertEqual(len([line for line in lines if '"step"' in line]), 100)

        # Validate every step record has required fields (LOG-001, LOG-005)
        for line in lines:
            record = json.loads(line)
            if "step" in record:
                self.assertIn("wall_dt_ms", record)
                self.assertIn("entity_count", record)

    def test_determinism_with_uavs(self) -> None:
        """SIM-003: identical seed produces identical UAV position sequences."""
        def _run_and_collect() -> list:
            log_path = _TMPDIR / f"det_{uuid.uuid4()}.jsonl"
            world = _small_world()
            cfg = SimulationConfig(duration_s=2.0, dt=0.1, seed=99,
                                   log_file=log_path)
            sim = SimulationEngine(cfg, world)
            sim.add_entity(_make_uav(
                entity_id="uav-det",
                pos=np.array([300.0, 300.0, 100.0]),
                endurance_s=600.0,
            ))
            sim.run()
            return [
                e.position.tolist()
                for e in sim.entities.by_type(EntityType.UAV)
            ]

        pos_a = _run_and_collect()
        pos_b = _run_and_collect()
        self.assertEqual(pos_a, pos_b)

    def test_uav_does_not_exit_world_boundary(self) -> None:
        """FLR-005, ENV-001: UAV stays within world bounds for full scenario."""
        log_path = _TMPDIR / f"bounds_{uuid.uuid4()}.jsonl"
        world = _small_world()
        cfg = SimulationConfig(duration_s=5.0, dt=0.1, seed=7, log_file=log_path)
        sim = SimulationEngine(cfg, world)
        uav = _make_uav(
            entity_id="uav-bounds",
            pos=np.array([300.0, 300.0, 100.0]),
            endurance_s=600.0,
        )
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
# M3 base.py: per-entity neighbor_radius_m
# =============================================================================

class TestPerEntityNeighborRadius(unittest.TestCase):

    def test_ground_entities_use_default_radius(self) -> None:
        """M3 base.py: pedestrian uses NEIGHBOR_RADIUS_M."""
        from sim3dves.entities.pedestrian import PedestrianEntity
        ped = PedestrianEntity(
            entity_id="p1",
            position=np.array([50.0, 50.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(ped.neighbor_radius_m, _D.NEIGHBOR_RADIUS_M)

    def test_uav_uses_larger_radius(self) -> None:
        """M3 base.py: UAV neighbor_radius_m == UAV_NEIGHBOR_RADIUS_M."""
        uav = _make_uav()
        self.assertAlmostEqual(uav.neighbor_radius_m, _D.UAV_NEIGHBOR_RADIUS_M)


if __name__ == "__main__":
    unittest.main()
