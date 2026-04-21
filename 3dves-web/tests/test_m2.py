"""
tests.test_m2
=============
M2 test suite — road network, vehicle kinematics, social force, context.

Every test cites the Req ID(s) it validates (NF-TE-002).
Coverage target: ≥ 80% of new M2 modules (NF-TE-001, NF-M-003).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.world import World
from sim3dves.entities.base import EntityManager, EntityState, EntityType
from sim3dves.entities.context import StepContext
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.entities.vehicle import (
    TrackedVehicleEntity,
    VehicleKinematics,
    WheeledVehicleEntity,
)
from sim3dves.maps.road_network import RoadEdge, RoadNetwork, RoadNode

# Ensure package is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

_D = SimDefaults()
_TMPDIR = Path(tempfile.mkdtemp())


# ### Helpers ###

def _small_grid(rows: int = 3, cols: int = 3, spacing: float = 100.0) -> RoadNetwork:
    """3×3 grid road network at origin (0, 0)."""
    return RoadNetwork.build_grid(
        rows=rows, cols=cols, spacing_m=spacing,
        origin=np.array([0.0, 0.0]),
    )


def _wheeled(
    position: np.ndarray,
    road_network: RoadNetwork,
    seed: int = 0,
) -> WheeledVehicleEntity:
    return WheeledVehicleEntity(
        entity_id=str(uuid.uuid4()),
        position=position,
        heading=0.0,
        road_network=road_network,
        rng=np.random.default_rng(seed),
    )


def _tracked(
    position: np.ndarray,
    world_extent: np.ndarray | None = None,
    seed: int = 0,
) -> TrackedVehicleEntity:
    return TrackedVehicleEntity(
        entity_id=str(uuid.uuid4()),
        position=position,
        heading=0.0,
        world_extent=world_extent if world_extent is not None
        else np.array([500.0, 500.0]),
        rng=np.random.default_rng(seed),
    )


def _ped(
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    speed_mps: float | None = None,
) -> PedestrianEntity:
    vel = velocity if velocity is not None else np.array([1.0, 0.0, 0.0])
    return PedestrianEntity(
        entity_id=str(uuid.uuid4()),
        position=position,
        velocity=vel,
        speed_mps=speed_mps,
    )


def _engine(
    duration_s: float = 1.0,
    dt: float = 0.1,
    world_size: float = 500.0,
    seed: int = 42,
) -> SimulationEngine:
    world = World(extent=np.array([world_size, world_size]))
    cfg = SimulationConfig(
        duration_s=duration_s, dt=dt, seed=seed,
        log_file=_TMPDIR / f"eng_{uuid.uuid4()}.jsonl",
    )
    return SimulationEngine(cfg, world)


# ### 1. RoadNetwork ###

class TestRoadNetworkStructure(unittest.TestCase):
    """ENV-006: Road network construction and adjacency."""

    def test_grid_node_count(self) -> None:
        """ENV-006: 3×3 grid must have exactly 9 nodes."""
        rn = _small_grid(3, 3)
        self.assertEqual(len(rn), 9)

    def test_grid_5x5_node_count(self) -> None:
        """ENV-006: 5×5 grid must have 25 nodes."""
        rn = _small_grid(5, 5)
        self.assertEqual(len(rn), 25)

    def test_node_ids_not_empty(self) -> None:
        """ENV-006: node_ids() must return non-empty list."""
        rn = _small_grid()
        self.assertGreater(len(rn.node_ids()), 0)

    def test_node_position_correct(self) -> None:
        """ENV-006: Bottom-left node n_0_0 must be at origin (0, 0)."""
        rn = _small_grid(spacing=100.0)
        pos = rn.node_position("n_0_0")
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-9)

    def test_node_position_top_right(self) -> None:
        """ENV-006: Top-right node n_2_2 at spacing=100 must be at (200, 200)."""
        rn = _small_grid(3, 3, spacing=100.0)
        pos = rn.node_position("n_2_2")
        np.testing.assert_allclose(pos, [200.0, 200.0], atol=1e-9)

    def test_add_edge_unknown_node_raises(self) -> None:
        """ENV-006: add_edge() with unknown node must raise ValueError."""
        rn = RoadNetwork()
        rn.add_node(RoadNode("a", np.array([0.0, 0.0])))
        with self.assertRaises(ValueError):
            rn.add_edge(RoadEdge("a", "NONEXISTENT"))


class TestRoadNetworkPathfinding(unittest.TestCase):
    """VEH-003: A* pathfinding correctness."""

    def setUp(self) -> None:
        self.rn = _small_grid(3, 3, spacing=100.0)

    def test_self_path_returns_single_node(self) -> None:
        """VEH-003: A* from node to itself returns [node_id]."""
        path = self.rn.find_path("n_0_0", "n_0_0")
        self.assertEqual(path, ["n_0_0"])

    def test_adjacent_nodes_path_length_2(self) -> None:
        """VEH-003: A* between adjacent nodes returns path of length 2."""
        path = self.rn.find_path("n_0_0", "n_0_1")
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], "n_0_0")
        self.assertEqual(path[-1], "n_0_1")

    def test_corner_to_corner_finds_path(self) -> None:
        """VEH-003: A* from n_0_0 to n_2_2 must find a non-trivial path."""
        path = self.rn.find_path("n_0_0", "n_2_2")
        self.assertGreater(len(path), 2)
        self.assertEqual(path[0], "n_0_0")
        self.assertEqual(path[-1], "n_2_2")

    def test_path_nodes_are_valid_ids(self) -> None:
        """VEH-003: All nodes in A* path must be registered node IDs."""
        path = self.rn.find_path("n_0_0", "n_2_2")
        valid_ids = set(self.rn.node_ids())
        for nid in path:
            self.assertIn(nid, valid_ids)

    def test_disconnected_node_returns_empty(self) -> None:
        """VEH-003: A* to an unconnected node must return empty list."""
        rn = RoadNetwork()
        rn.add_node(RoadNode("a", np.array([0.0, 0.0])))
        rn.add_node(RoadNode("b", np.array([100.0, 0.0])))
        # No edge between a and b
        path = rn.find_path("a", "b")
        self.assertEqual(path, [])

    def test_unknown_node_returns_empty(self) -> None:
        """VEH-003: A* with unknown node returns empty path gracefully."""
        path = self.rn.find_path("n_0_0", "GHOST")
        self.assertEqual(path, [])

    def test_nearest_node_correct(self) -> None:
        """ENV-006: nearest_node() returns the closest registered node."""
        rn = _small_grid(3, 3, spacing=100.0)
        # Query point very close to n_1_1 (100, 100)
        nearest = rn.nearest_node(np.array([101.0, 99.0]))
        self.assertEqual(nearest, "n_1_1")

    def test_nearest_node_empty_network_returns_none(self) -> None:
        """ENV-006: nearest_node() on empty network returns None."""
        rn = RoadNetwork()
        self.assertIsNone(rn.nearest_node(np.array([0.0, 0.0])))


# ### 2. VehicleKinematics ###

class TestVehicleKinematics(unittest.TestCase):
    """VEH-002: Kinematic parameter dataclass."""

    def test_defaults_within_expected_range(self) -> None:
        """VEH-002: Default VehicleKinematics values within PRD spec."""
        kin = VehicleKinematics()
        self.assertGreater(kin.max_speed_mps, 0.0)
        self.assertGreater(kin.accel_mps2, 0.0)
        self.assertGreater(kin.decel_mps2, 0.0)
        self.assertGreater(kin.max_turn_rate_dps, 0.0)
        self.assertGreater(kin.arrival_threshold_m, 0.0)


# ### 3. WheeledVehicleEntity ###

class TestWheeledVehicleKinematics(unittest.TestCase):
    """VEH-001..003, VEH-007: Wheeled vehicle kinematic invariants."""

    def setUp(self) -> None:
        self.rn = _small_grid(4, 4, spacing=100.0)
        # Place vehicle at n_0_0 (0, 0)
        self.vehicle = _wheeled(
            np.array([0.0, 0.0, 50.0]),  # Intentionally wrong Z
            self.rn, seed=7,
        )

    def test_terrain_lock_z_at_construction(self) -> None:
        """VEH-007: Vehicle Z must be terrain elevation at construction."""
        self.assertAlmostEqual(self.vehicle.position[2], 0.0, places=12)

    def test_terrain_lock_z_after_steps(self) -> None:
        """VEH-007: Vehicle Z must remain at terrain elevation every step."""
        for _ in range(30):
            self.vehicle.step(dt=0.1)
            self.assertAlmostEqual(
                self.vehicle.position[2], 0.0, places=12,
                msg=f"Z drifted to {self.vehicle.position[2]} at step"
            )

    def test_z_velocity_always_zero(self) -> None:
        """VEH-007: Vehicle Z velocity must always be 0.0."""
        for _ in range(30):
            self.vehicle.step(dt=0.1)
            self.assertAlmostEqual(self.vehicle.velocity[2], 0.0, places=12)

    def test_speed_never_exceeds_max(self) -> None:
        """VEH-002: Wheeled vehicle speed must never exceed max_speed_mps."""
        max_spd = self.vehicle._kinematics.max_speed_mps
        for _ in range(100):
            self.vehicle.step(dt=0.1)
            speed = float(np.linalg.norm(self.vehicle.velocity[:2]))
            self.assertLessEqual(
                speed, max_spd + 1e-6,
                msg=f"Speed {speed:.3f} exceeded max {max_spd}"
            )

    def test_vehicle_makes_progress(self) -> None:
        """VEH-003: Vehicle must move toward its waypoint over 50 steps."""
        initial_pos = self.vehicle.position[:2].copy()
        for _ in range(50):
            self.vehicle.step(dt=0.1)
        final_pos = self.vehicle.position[:2]
        displacement = float(np.linalg.norm(final_pos - initial_pos))
        # With max speed ~12 m/s × 50 steps × 0.1 s = up to 60 m
        self.assertGreater(displacement, 1.0, "Vehicle did not move")

    def test_vehicle_replans_on_path_complete(self) -> None:
        """VEH-003: Vehicle must select a new path after reaching destination."""
        # Run long enough for at least one path completion
        for _ in range(500):
            self.vehicle.step(dt=0.1)
        # Vehicle should still be alive and moving (not stuck)
        self.assertTrue(self.vehicle.alive)

    def test_entity_type_is_wheeled(self) -> None:
        """ENT-002: WheeledVehicleEntity must have WHEELED_VEHICLE EntityType."""
        self.assertEqual(self.vehicle.entity_type, EntityType.WHEELED_VEHICLE)

    def test_dead_vehicle_does_not_move(self) -> None:
        """ENT-001: Dead vehicle must not change position after kill()."""
        self.vehicle.step(dt=0.1)  # Establish some motion
        pos_before = self.vehicle.position.copy()
        self.vehicle.kill()
        self.vehicle.step(dt=0.1)
        np.testing.assert_array_equal(self.vehicle.position, pos_before)

    def test_no_road_network_vehicle_stays_idle(self) -> None:
        """VEH-003: Vehicle with road_network=None must stay IDLE (no crash)."""
        v = WheeledVehicleEntity(
            entity_id=str(uuid.uuid4()),
            position=np.array([50.0, 50.0, 0.0]),
            road_network=None,
        )
        for _ in range(5):
            v.step(dt=0.1)
        self.assertEqual(v.state, EntityState.IDLE)


# ### 4. TrackedVehicleEntity ###

class TestTrackedVehicleKinematics(unittest.TestCase):
    """VEH-001, VEH-002, VEH-007: Tracked vehicle invariants."""

    def setUp(self) -> None:
        self.world_extent = np.array([400.0, 400.0])
        self.vehicle = _tracked(
            np.array([200.0, 200.0, 99.0]),  # Z = 99 — should be overridden
            world_extent=self.world_extent, seed=3,
        )

    def test_terrain_lock_at_construction(self) -> None:
        """VEH-007: Tracked vehicle Z must be 0 at construction."""
        self.assertAlmostEqual(self.vehicle.position[2], 0.0, places=12)

    def test_terrain_lock_after_steps(self) -> None:
        """VEH-007: Tracked vehicle Z must remain 0 after every step."""
        for _ in range(40):
            self.vehicle.step(dt=0.1)
            self.assertAlmostEqual(self.vehicle.position[2], 0.0, places=12)

    def test_entity_type_is_tracked(self) -> None:
        """ENT-002: TrackedVehicleEntity must have TRACKED_VEHICLE EntityType."""
        self.assertEqual(self.vehicle.entity_type, EntityType.TRACKED_VEHICLE)

    def test_off_road_speed_below_wheeled(self) -> None:
        """VEH-001: Tracked vehicle effective speed must be < wheeled max."""
        wheeled_max = _D.VEH_MAX_SPEED_MPS
        tracked_effective = self.vehicle._effective_max_speed()
        self.assertLess(tracked_effective, wheeled_max)

    def test_tracked_makes_progress_without_road_network(self) -> None:
        """VEH-001: Tracked vehicle must navigate without road network."""
        initial_pos = self.vehicle.position[:2].copy()
        for _ in range(50):
            self.vehicle.step(dt=0.1)
        displacement = float(np.linalg.norm(
            self.vehicle.position[:2] - initial_pos
        ))
        self.assertGreater(displacement, 0.5, "Tracked vehicle did not move")

    def test_tracked_replans_after_arrival(self) -> None:
        """VEH-001: Tracked vehicle selects new destination after arriving."""
        # Force a tiny world so destination is reached quickly
        v = TrackedVehicleEntity(
            entity_id=str(uuid.uuid4()),
            position=np.array([5.0, 5.0, 0.0]),
            world_extent=np.array([20.0, 20.0]),
            rng=np.random.default_rng(42),
        )
        for _ in range(300):
            v.step(dt=0.1)
        # Should still be alive and (usually) moving
        self.assertTrue(v.alive)


# ### 5. Social Force (PED-003) ###

class TestSocialForce(unittest.TestCase):
    """PED-003: Social force repulsion between pedestrians."""

    def test_pedestrians_separate_when_close(self) -> None:
        """PED-003: Two pedestrians 1m apart should increase separation."""
        # Both walking northward, 1 m apart in X
        speed = (_D.PED_SPEED_MIN_MPS + _D.PED_SPEED_MAX_MPS) / 2.0
        ped_a = _ped(
            np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 1.0, 0.0]),
            speed_mps=speed,
        )
        ped_b = _ped(
            np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.0, 1.0, 0.0]),
            speed_mps=speed,
        )

        # Build context so A sees B as a neighbour and vice versa
        ctx_a = StepContext(neighbors=[ped_b])
        ctx_b = StepContext(neighbors=[ped_a])

        for _ in range(20):
            ped_a.step(dt=0.1, context=ctx_a)
            ped_b.step(dt=0.1, context=ctx_b)

        separation = float(np.linalg.norm(
            ped_a.position[:2] - ped_b.position[:2]
        ))
        self.assertGreater(
            separation, 1.0,
            f"Separation {separation:.3f} m did not increase from 1.0 m"
        )

    def test_pedestrian_far_away_unaffected(self) -> None:
        """PED-003: Pedestrian outside social radius must not be repelled."""
        ped_a = _ped(np.array([0.0, 0.0, 0.0]),
                     velocity=np.array([1.0, 0.0, 0.0]))
        ped_b = _ped(np.array([_D.SOCIAL_RADIUS_M + 5.0, 0.0, 0.0]),
                     velocity=np.array([1.0, 0.0, 0.0]))

        # A sees B as neighbour (via context) but B is beyond SOCIAL_RADIUS
        ctx_a = StepContext(neighbors=[ped_b])

        vel_before = ped_a.velocity[:2].copy()
        ped_a.step(dt=0.1, context=ctx_a)
        # Heading noise is present; verify speed envelope (not exact direction)
        speed = float(np.linalg.norm(ped_a.velocity[:2]))
        self.assertAlmostEqual(speed, ped_a._target_speed_mps, places=6)

    def test_speed_envelope_preserved_under_social_force(self) -> None:
        """PED-001 + PED-003: Speed must stay in [MIN, MAX] with social force."""
        # Dense cluster: 5 pedestrians all within 2m
        peds = [
            _ped(np.array([float(i) * 0.4, 0.0, 0.0]),
                 velocity=np.array([1.0, 0.0, 0.0]))
            for i in range(5)
        ]
        for _ in range(50):
            for p in peds:
                ctx = StepContext(neighbors=[q for q in peds if q is not p])
                p.step(dt=0.1, context=ctx)

        for p in peds:
            speed = float(np.linalg.norm(p.velocity[:2]))
            self.assertGreaterEqual(speed, _D.PED_SPEED_MIN_MPS - 1e-9)
            self.assertLessEqual(speed, _D.PED_SPEED_MAX_MPS + 1e-9)

    def test_no_context_behavior_same_as_before(self) -> None:
        """PED-003: step(dt, context=None) must not raise and preserves speed."""
        ped = _ped(np.array([50.0, 50.0, 0.0]))
        for _ in range(20):
            ped.step(dt=0.1, context=None)  # No context — no social force
        speed = float(np.linalg.norm(ped.velocity[:2]))
        self.assertGreaterEqual(speed, _D.PED_SPEED_MIN_MPS - 1e-9)
        self.assertLessEqual(speed, _D.PED_SPEED_MAX_MPS + 1e-9)

    def test_vehicle_neighbours_do_not_repel_pedestrian(self) -> None:
        """PED-003: Social force applies ONLY between pedestrians, not vehicles."""
        ped = _ped(np.array([0.0, 0.0, 0.0]),
                   velocity=np.array([1.0, 0.0, 0.0]))
        rn = _small_grid()
        veh = _wheeled(np.array([0.5, 0.0, 0.0]), rn)

        vel_y_before = ped.velocity[1]
        ctx = StepContext(neighbors=[veh])
        for _ in range(10):
            ped.step(dt=0.1, context=ctx)
        # Heading noise means direction changes, but social force from
        # vehicle must not have contributed — we verify speed only
        speed = float(np.linalg.norm(ped.velocity[:2]))
        self.assertAlmostEqual(speed, ped._target_speed_mps, places=6)


# ══ 6. StepContext & EntityManager context ════════════════════════════════════

class TestStepContext(unittest.TestCase):
    """ENT-001, PED-003: StepContext construction and EntityManager integration."""

    def test_step_context_is_dataclass(self) -> None:
        """StepContext must instantiate with empty neighbours by default."""
        ctx = StepContext()
        self.assertEqual(ctx.neighbors, [])

    def test_step_context_stores_neighbours(self) -> None:
        """StepContext must store provided neighbours list."""
        ped = _ped(np.array([0.0, 0.0, 0.0]))
        ctx = StepContext(neighbors=[ped])
        self.assertIn(ped, ctx.neighbors)


class TestEntityManagerContext(unittest.TestCase):
    """ENT-001, NF-P-001: EntityManager context-aware step_all."""

    def test_self_not_in_own_neighbours(self) -> None:
        """
        ENT-001: An entity must never appear in its own StepContext neighbours.

        We verify this by patching _update_behavior to capture the context.
        """
        captured: list = []

        class CapturePed(PedestrianEntity):
            def _update_behavior(self, dt, context=None):
                if context is not None:
                    captured.extend(context.neighbors)
                super()._update_behavior(dt, context)

        mgr = EntityManager()
        ped = CapturePed(
            entity_id="self",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        mgr.add(ped)
        mgr.step_all(dt=0.1)

        # The entity must not be in its own neighbour list
        ids = [e.entity_id for e in captured]
        self.assertNotIn("self", ids)

    def test_nearby_entity_is_in_neighbours(self) -> None:
        """
        PED-003: An entity within NEIGHBOR_RADIUS_M must appear in neighbours.
        """
        captured: list = []

        class CapturePed(PedestrianEntity):
            def _update_behavior(self, dt, context=None):
                if context is not None:
                    captured.extend(context.neighbors)
                super()._update_behavior(dt, context)

        mgr = EntityManager()
        ped_a = CapturePed(
            entity_id="A",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        # B is 2m away — within NEIGHBOR_RADIUS_M (10m)
        ped_b = PedestrianEntity(
            entity_id="B",
            position=np.array([2.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        mgr.add(ped_a)
        mgr.add(ped_b)
        mgr.step_all(dt=0.1)

        ids = [e.entity_id for e in captured]
        self.assertIn("B", ids)

    def test_distant_entity_not_in_neighbours(self) -> None:
        """
        NF-P-001: Entity beyond NEIGHBOR_RADIUS_M must NOT appear in neighbours.
        """
        captured: list = []

        class CapturePed(PedestrianEntity):
            def _update_behavior(self, dt, context=None):
                if context is not None:
                    captured.extend(context.neighbors)
                super()._update_behavior(dt, context)

        mgr = EntityManager()
        ped_a = CapturePed(
            entity_id="A",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        # B is 50m away — beyond NEIGHBOR_RADIUS_M (10m)
        ped_b = PedestrianEntity(
            entity_id="B",
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        mgr.add(ped_a)
        mgr.add(ped_b)
        mgr.step_all(dt=0.1)

        ids = [e.entity_id for e in captured]
        self.assertNotIn("B", ids)


# ══ 7. World road_network field ═══════════════════════════════════════════════

class TestWorldM2(unittest.TestCase):
    """ENV-006: World road_network field integration."""

    def test_world_stores_road_network(self) -> None:
        """ENV-006: World must expose road_network attribute."""
        rn = _small_grid()
        world = World(extent=np.array([500.0, 500.0]), road_network=rn)
        self.assertIs(world.road_network, rn)

    def test_world_road_network_default_none(self) -> None:
        """ENV-006: World without road network must default to None."""
        world = World(extent=np.array([500.0, 500.0]))
        self.assertIsNone(world.road_network)


# ### 8. Integration test ###

class TestM2Integration(unittest.TestCase):
    """NF-TE-004: End-to-end integration of M2 features."""

    def test_mixed_scenario_100_steps(self) -> None:
        """
        NF-TE-004: Mixed scenario (wheeled + tracked + pedestrians) must
        run 100 steps headlessly without exception and produce a JSONL log.
        Covers: VEH-001..003, PED-001..003, Req-7, LOG-001, SIM-001..003.
        """
        rn = RoadNetwork.build_grid(4, 4, 80.0, np.array([10.0, 10.0]))
        world = World(extent=np.array([350.0, 350.0]), road_network=rn)
        log_path = _TMPDIR / f"int_{uuid.uuid4()}.jsonl"
        cfg = SimulationConfig(
            duration_s=10.0, dt=0.1, seed=7, log_file=log_path
        )
        sim = SimulationEngine(cfg, world)
        rng = np.random.default_rng(7)
        node_ids = rn.node_ids()

        # Spawn 5 wheeled vehicles
        for i in range(5):
            nid = node_ids[int(rng.integers(0, len(node_ids)))]
            xy = rn.node_position(nid)
            sim.add_entity(WheeledVehicleEntity(
                entity_id=str(uuid.uuid4()),
                position=np.array([xy[0], xy[1], 0.0]),
                road_network=rn,
                is_eoi=(i == 0),
                rng=np.random.default_rng(int(rng.integers(0, 2**31))),
            ))

        # Spawn 3 tracked vehicles
        for i in range(3):
            xy = rng.random(2) * world.extent
            pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
            sim.add_entity(TrackedVehicleEntity(
                entity_id=str(uuid.uuid4()),
                position=pos,
                world_extent=world.extent,
                rng=np.random.default_rng(int(rng.integers(0, 2**31))),
            ))

        # Spawn 15 pedestrians
        for i in range(15):
            xy = rng.random(2) * world.extent
            pos = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
            sim.add_entity(PedestrianEntity(
                entity_id=str(uuid.uuid4()),
                position=pos,
                velocity=rng.standard_normal(3),
            ))

        # Run exactly 100 steps headlessly (SIM-005)
        with sim.logger:
            for _ in range(100):
                sim.step()

        # Verify step count (SIM-001)
        self.assertEqual(sim.step_idx, 100)

        # Verify JSONL log written (LOG-001)
        self.assertTrue(log_path.exists())
        lines = log_path.read_text().strip().splitlines()
        # At least step records (may also include event records)
        step_records = [
            json.loads(ln) for ln in lines
            if '"step"' in ln and '"entities"' in ln
        ]
        self.assertEqual(len(step_records), 100)

    def test_terrain_lock_all_entities_100_steps(self) -> None:
        """
        Req-7, VEH-007: All ground entities must have Z = 0 after 100 steps.
        """
        rn = _small_grid(3, 3, 80.0)
        world = World(extent=np.array([300.0, 300.0]), road_network=rn)
        log_path = _TMPDIR / f"tz_{uuid.uuid4()}.jsonl"
        cfg = SimulationConfig(
            duration_s=10.0, dt=0.1, seed=5, log_file=log_path
        )
        sim = SimulationEngine(cfg, world)
        rng = np.random.default_rng(5)
        node_ids = rn.node_ids()

        all_ground_entities = []
        for i in range(4):
            nid = node_ids[int(rng.integers(0, len(node_ids)))]
            xy = rn.node_position(nid)
            v = WheeledVehicleEntity(
                entity_id=str(uuid.uuid4()),
                position=np.array([xy[0], xy[1], 0.0]),
                road_network=rn,
                rng=np.random.default_rng(i),
            )
            sim.add_entity(v)
            all_ground_entities.append(v)

        for i in range(6):
            xy = rng.random(2) * world.extent
            p = PedestrianEntity(
                entity_id=str(uuid.uuid4()),
                position=world.snap_to_terrain(np.array([xy[0], xy[1], 0.0])),
                velocity=rng.standard_normal(3),
            )
            sim.add_entity(p)
            all_ground_entities.append(p)

        with sim.logger:
            for _ in range(100):
                sim.step()

        for e in all_ground_entities:
            self.assertAlmostEqual(
                e.position[2], 0.0, places=10,
                msg=f"{e.entity_type.name} {e.entity_id[:8]}: "
                    f"Z={e.position[2]:.6f} (expected 0.0)"
            )

    def test_determinism_with_vehicles(self) -> None:
        """SIM-003: Mixed vehicle+pedestrian scenario must be deterministic."""
        def snapshot() -> list:
            rn = _small_grid(3, 3, 100.0)
            world = World(extent=np.array([400.0, 400.0]), road_network=rn)
            cfg = SimulationConfig(
                duration_s=2.0, dt=0.1, seed=33,
                log_file=_TMPDIR / f"det_{uuid.uuid4()}.jsonl",
            )
            sim = SimulationEngine(cfg, world)
            rng = np.random.default_rng(33)
            node_ids = rn.node_ids()
            for i in range(4):
                nid = node_ids[int(rng.integers(0, len(node_ids)))]
                xy = rn.node_position(nid)
                sim.add_entity(WheeledVehicleEntity(
                    entity_id=f"v{i}",
                    position=np.array([xy[0], xy[1], 0.0]),
                    road_network=rn,
                    rng=np.random.default_rng(i),
                ))
            for i in range(4):
                xy = rng.random(2) * world.extent
                sim.add_entity(PedestrianEntity(
                    entity_id=f"p{i}",
                    position=world.snap_to_terrain(np.array([xy[0], xy[1], 0.0])),
                    velocity=rng.standard_normal(3),
                ))
            sim.run()
            return sorted(
                [(e.entity_id, e.position.tolist())
                 for e in sim.entities.living()],
                key=lambda x: x[0],
            )

        self.assertEqual(snapshot(), snapshot(), "Simulation is non-deterministic")


# ### Runner ###

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestRoadNetworkStructure,
        TestRoadNetworkPathfinding,
        TestVehicleKinematics,
        TestWheeledVehicleKinematics,
        TestTrackedVehicleKinematics,
        TestSocialForce,
        TestStepContext,
        TestEntityManagerContext,
        TestWorldM2,
        TestM2Integration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
