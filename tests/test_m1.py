"""
tests.test_smoke
================
M1 smoke and unit test suite.

Every test function references the requirement(s) it validates in its
docstring.  At least one test per functional requirement (NF-TE-001,
NF-M-003).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-TE-001: Unit tests at all levels included at this release.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.engine import SimulationConfig, SimulationEngine
from sim3dves.core.event_bus import Event, EventBus, EventType
from sim3dves.core.world import AABB, NFZCylinder, World
from sim3dves.entities.base import EntityManager, EntityState, EntityType
from sim3dves.entities.pedestrian import PedestrianEntity
from sim3dves.logging.logger import Logger

_D = SimDefaults()


# ### Fixtures ###

@pytest.fixture
def small_world() -> World:
    """100 × 100 m flat world with no structures or NFZs."""
    return World(extent=np.array([100.0, 100.0]))


@pytest.fixture
def sim(small_world: World, tmp_path: Path) -> SimulationEngine:
    """Minimal engine (1 s / 0.1 s dt) with a temp log file."""
    cfg = SimulationConfig(
        duration_s=1.0,
        dt=0.1,
        seed=42,
        log_file=tmp_path / "test.jsonl",
    )
    return SimulationEngine(cfg, small_world)


def _spawn_pedestrian(
    rng: np.random.Generator,
    world: World,
    is_eoi: bool = False,
) -> PedestrianEntity:
    """Factory helper: create a terrain-snapped pedestrian with random XY."""
    xy = rng.random(2) * world.extent
    position = world.snap_to_terrain(np.array([xy[0], xy[1], 0.0]))
    velocity = rng.standard_normal(3)
    return PedestrianEntity(
        entity_id=str(uuid.uuid4()),
        position=position,
        velocity=velocity,
        is_eoi=is_eoi,
    )


# ### ENT-001 / ENT-002: Entity base model and type enum ###

class TestEntityBase:
    """ENT-001, ENT-002: Entity model and EntityType enum."""

    def test_entity_type_is_enum(self) -> None:
        """ENT-002: EntityType must be an Enum, not a raw string."""
        assert isinstance(EntityType.PEDESTRIAN, EntityType)
        assert isinstance(EntityType.UAV, EntityType)
        assert isinstance(EntityType.WHEELED_VEHICLE, EntityType)
        assert isinstance(EntityType.TRACKED_VEHICLE, EntityType)

    def test_entity_state_is_enum(self) -> None:
        """ENT-003: EntityState must be an Enum."""
        assert isinstance(EntityState.IDLE, EntityState)
        assert isinstance(EntityState.MOVING, EntityState)
        assert isinstance(EntityState.DEAD, EntityState)

    def test_kill_sets_alive_false_and_dead_state(self, small_world: World) -> None:
        """ENT-001: kill() must set alive=False and state=DEAD."""
        ped = PedestrianEntity(
            entity_id="kill-test",
            position=np.array([50.0, 50.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        ped.kill()
        assert not ped.alive
        assert ped.state == EntityState.DEAD

    def test_dead_entity_does_not_move(self) -> None:
        """ENT-001: Dead entity position must not change after kill()."""
        ped = PedestrianEntity(
            entity_id="frozen",
            position=np.array([10.0, 10.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        ped.kill()
        pos_before = ped.position.copy()
        ped.step(dt=1.0)
        np.testing.assert_array_equal(ped.position, pos_before)


# ### PED-001: Pedestrian speed envelope ###

class TestPedestrianKinematics:
    """PED-001, PED-004, Req-7: Pedestrian kinematic invariants."""

    def test_speed_within_envelope_after_many_steps(
        self, small_world: World
    ) -> None:
        """PED-001: XY speed must remain in [PED_SPEED_MIN, PED_SPEED_MAX]."""
        rng = np.random.default_rng(0)
        for _ in range(30):
            ped = _spawn_pedestrian(rng, small_world)
            for _ in range(50):
                ped.step(dt=0.1)
            speed = float(np.linalg.norm(ped.velocity[:2]))
            assert _D.PED_SPEED_MIN_MPS - 1e-9 <= speed <= _D.PED_SPEED_MAX_MPS + 1e-9, (
                f"Speed {speed:.4f} outside envelope "
                f"[{_D.PED_SPEED_MIN_MPS}, {_D.PED_SPEED_MAX_MPS}]"
            )

    def test_z_velocity_always_zero(self, small_world: World) -> None:
        """Req-7: Pedestrian Z velocity must always be 0.0 (terrain lock)."""
        rng = np.random.default_rng(1)
        ped = _spawn_pedestrian(rng, small_world)
        for _ in range(30):
            ped.step(dt=0.1)
            assert ped.velocity[2] == pytest.approx(0.0, abs=1e-12), (
                f"Z velocity leaked: vz={ped.velocity[2]}"
            )

    def test_z_position_on_terrain(self, small_world: World) -> None:
        """Req-7: Pedestrian Z position must equal terrain elevation after every step."""
        rng = np.random.default_rng(2)
        ped = _spawn_pedestrian(rng, small_world)
        terrain_z = small_world.terrain_elevation(ped.position[:2])
        for _ in range(30):
            ped.step(dt=0.1)
            assert ped.position[2] == pytest.approx(terrain_z, abs=1e-12), (
                f"Z drifted off terrain: z={ped.position[2]}, expected {terrain_z}"
            )

    def test_eoi_flag_persists(self) -> None:
        """PED-004: is_eoi flag must be readable and persist across steps."""
        ped = PedestrianEntity(
            entity_id="eoi-check",
            position=np.array([10.0, 10.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            is_eoi=True,
        )
        for _ in range(5):
            ped.step(dt=0.1)
        assert ped.is_eoi is True

    def test_heading_derived_from_velocity(self) -> None:
        """ENT-001: Heading must equal atan2(vy, vx) after step."""
        ped = PedestrianEntity(
            entity_id="head-check",
            position=np.array([50.0, 50.0, 0.0]),
            velocity=np.array([1.0, 1.0, 0.0]),  # 45-degree direction
        )
        ped.step(dt=0.1)
        expected = math.degrees(math.atan2(ped.velocity[1], ped.velocity[0]))
        assert ped.heading == pytest.approx(expected, abs=1.0)


# ### EntityManager ###

class TestEntityManager:
    """ENT-001: EntityManager registry behaviour."""

    def test_duplicate_id_raises(self) -> None:
        """EntityManager must raise ValueError on duplicate entity_id."""
        mgr = EntityManager()
        ped = PedestrianEntity(
            entity_id="dup",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        mgr.add(ped)
        with pytest.raises(ValueError, match="Duplicate entity_id"):
            mgr.add(ped)

    def test_living_excludes_dead(self) -> None:
        """living() must not include entities where alive=False."""
        mgr = EntityManager()
        alive_ped = PedestrianEntity(
            entity_id="alive",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        dead_ped = PedestrianEntity(
            entity_id="dead",
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        dead_ped.kill()
        mgr.add(alive_ped)
        mgr.add(dead_ped)
        assert len(mgr.living()) == 1
        assert mgr.living()[0].entity_id == "alive"

    def test_by_type_filter(self) -> None:
        """by_type() must return only entities matching the requested EntityType."""
        mgr = EntityManager()
        ped = PedestrianEntity(
            entity_id="ped1",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        mgr.add(ped)
        pedestrians = mgr.by_type(EntityType.PEDESTRIAN)
        uavs = mgr.by_type(EntityType.UAV)
        assert len(pedestrians) == 1
        assert len(uavs) == 0


# ### World ###

class TestWorld:
    """ENV-001 to ENV-007: World spatial model."""

    def test_in_bounds_interior(self) -> None:
        """ENV-001: In-bounds check must return True for interior positions."""
        world = World(extent=np.array([500.0, 500.0]))
        assert world.in_bounds(np.array([250.0, 250.0, 0.0]))

    def test_in_bounds_exterior(self) -> None:
        """ENV-001: In-bounds check must return False for exterior positions."""
        world = World(extent=np.array([500.0, 500.0]))
        assert not world.in_bounds(np.array([501.0, 250.0, 0.0]))

    def test_snap_to_terrain_zeroes_z(self) -> None:
        """Req-7: snap_to_terrain() must set Z to terrain elevation (0 for flat)."""
        world = World(extent=np.array([100.0, 100.0]))
        pos = np.array([30.0, 40.0, 99.0])   # Intentionally wrong Z
        snapped = world.snap_to_terrain(pos)
        assert snapped[2] == pytest.approx(0.0)
        assert snapped[0] == pytest.approx(30.0)  # XY must be preserved

    def test_nfz_cylinder_detection(self) -> None:
        """ENV-005: NFZ cylinder must correctly identify violations."""
        nfz = NFZCylinder(
            center_xy=np.array([100.0, 100.0]),
            radius_m=50.0,
            alt_max_m=200.0,
        )
        world = World(extent=np.array([500.0, 500.0]), nfz_cylinders=[nfz])
        inside = np.array([110.0, 100.0, 50.0])   # Within radius and altitude
        outside = np.array([200.0, 100.0, 50.0])  # Outside radius
        assert world.in_nfz(inside)
        assert not world.in_nfz(outside)

    def test_aabb_structure_detection(self) -> None:
        """ENV-003: AABB structure footprint must be detected correctly."""
        struct = AABB(x=10.0, y=10.0, width=20.0, depth=20.0, height=5.0)
        world = World(extent=np.array([100.0, 100.0]), structures=[struct])
        inside = np.array([15.0, 15.0, 0.0])
        outside = np.array([35.0, 15.0, 0.0])
        assert world.occluded_by_structure(inside)
        assert not world.occluded_by_structure(outside)


# ### EventBus ###

class TestEventBus:
    """SIM-002: Event bus publish-subscribe behaviour."""

    def test_subscriber_receives_event(self) -> None:
        """SIM-002: Subscriber must be called when matching event is published."""
        received: list = []
        bus = EventBus()
        bus.subscribe(EventType.OUT_OF_BOUNDS, lambda e: received.append(e))
        evt = Event(timestamp=1.0, event_type=EventType.OUT_OF_BOUNDS,
                    payload={"id": "x"})
        bus.publish(evt)
        assert len(received) == 1
        assert received[0].event_type == EventType.OUT_OF_BOUNDS

    def test_unsubscribe_stops_delivery(self) -> None:
        """EventBus.unsubscribe() must prevent further event delivery."""
        received: list = []
        handler = lambda e: received.append(e)  # noqa: E731
        bus = EventBus()
        bus.subscribe(EventType.DETECTION, handler)
        bus.unsubscribe(EventType.DETECTION, handler)
        bus.publish(Event(1.0, EventType.DETECTION))
        assert len(received) == 0

    def test_wrong_event_type_not_delivered(self) -> None:
        """EventBus must not deliver events to handlers for a different type."""
        received: list = []
        bus = EventBus()
        bus.subscribe(EventType.LOW_FUEL, lambda e: received.append(e))
        bus.publish(Event(1.0, EventType.DETECTION))
        assert len(received) == 0


# ### SimulationEngine ###

class TestSimulationEngine:
    """SIM-001, SIM-003, SIM-005, ENV-001: Engine-level behaviour."""

    def test_step_count_matches_duration(
        self, sim: SimulationEngine
    ) -> None:
        """SIM-001: step_idx after run() must equal floor(duration / dt)."""
        sim.run()
        expected = int(1.0 / 0.1)
        assert sim.step_idx == expected

    def test_sim_time_advances(self, sim: SimulationEngine) -> None:
        """SIM-001: sim_time must equal step_idx × dt after run()."""
        sim.run()
        assert sim.sim_time == pytest.approx(
            sim.step_idx * sim.config.dt, rel=1e-9
        )

    def test_out_of_bounds_kills_entity(
        self, sim: SimulationEngine
    ) -> None:
        """ENV-001: Entity leaving world bounds must be killed.

        NOTE: PedestrianEntity normalises XY speed to [PED_SPEED_MIN, PED_SPEED_MAX].
        With max speed 1.8 m/s and dt=0.1 s, each step moves at most 0.18 m.
        Placing the entity at X=99.9 with speed_mps=_D.PED_SPEED_MAX_MPS guarantees
        it exits the 100 m world within 1–2 steps regardless of heading noise.
        """
        ped = PedestrianEntity(
            entity_id="oob",
            # Place entity close enough to edge that even min speed exits in 1 step
            position=np.array([99.95, 50.0, 0.0]),
            # Pure +X velocity — normalised to speed_mps inside __init__
            velocity=np.array([1.0, 0.0, 0.0]),
            speed_mps=_D.PED_SPEED_MAX_MPS,   # Force max speed for determinism
        )
        sim.add_entity(ped)
        # Run enough steps; entity must exit within the first 2
        for _ in range(5):
            sim.step()
        assert not ped.alive, "Entity should have been killed on boundary exit"

    def test_determinism(self, small_world: World, tmp_path: Path) -> None:
        """SIM-003: Identical seed produces identical final positions."""
        def run_and_snapshot() -> list:
            cfg = SimulationConfig(
                duration_s=0.5,
                dt=0.1,
                seed=99,
                log_file=tmp_path / f"det_{uuid.uuid4()}.jsonl",
            )
            engine = SimulationEngine(cfg, small_world)
            rng = np.random.default_rng(99)
            for _ in range(8):
                engine.add_entity(_spawn_pedestrian(rng, small_world))
            engine.run()
            return [e.position.tolist() for e in engine.entities.living()]

        snap1 = run_and_snapshot()
        snap2 = run_and_snapshot()
        assert snap1 == snap2, "Simulation is non-deterministic"


# ### Logger ###

class TestLogger:
    """LOG-001, LOG-002, LOG-005: Logger output correctness."""

    def test_step_log_written(self, tmp_path: Path) -> None:
        """LOG-001: JSONL log must be written and contain step records."""
        log_path = tmp_path / "log.jsonl"
        ped = PedestrianEntity(
            entity_id="log-ped",
            position=np.array([10.0, 10.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        with Logger(log_path) as log:
            log.log_step(0, [ped], wall_dt_s=0.005)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 0
        assert "entities" in record
        assert record["entities"][0]["id"] == "log-ped"

    def test_perf_column_present(self, tmp_path: Path) -> None:
        """LOG-005: Step record must include wall_dt_ms performance column."""
        log_path = tmp_path / "perf.jsonl"
        ped = PedestrianEntity(
            entity_id="p",
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )
        with Logger(log_path) as log:
            log.log_step(0, [ped], wall_dt_s=0.012)
        record = json.loads(log_path.read_text())
        assert "wall_dt_ms" in record
        assert record["wall_dt_ms"] == pytest.approx(12.0, abs=0.1)

    def test_context_manager_closes_file(self, tmp_path: Path) -> None:
        """Logger must close the file handle on __exit__."""
        log_path = tmp_path / "cm.jsonl"
        with Logger(log_path) as log:
            pass
        assert log._file.closed, "File handle not closed after context manager exit"

    def test_event_log_written(self, tmp_path: Path) -> None:
        """LOG-002: log_event() must write a parseable JSONL record."""
        log_path = tmp_path / "evt.jsonl"
        with Logger(log_path) as log:
            log.log_event({"type": "TEST_EVENT", "value": 42})
        record = json.loads(log_path.read_text())
        assert record["type"] == "TEST_EVENT"
