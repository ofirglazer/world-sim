"""
sim3dves.entities.vehicle
=========================
Vehicle entities: WheeledVehicleEntity (A* road navigation) and
TrackedVehicleEntity (direct off-road navigation).

Design Patterns
---------------
* Template Method  : VehicleEntity inherits Entity.step() skeleton and
  provides concrete _update_behavior / _update_kinematics.  It adds its
  own abstract hook _plan_new_path() to subclass the path-planning strategy.
* Strategy (light) : path-planning strategy is fixed per concrete class rather
  than an injected object — sufficient for M2.

Physics model
-------------
Kinematic bicycle model (simplified):
  1. Compute angular error to next waypoint.
  2. Clamp turn to max_turn_rate_dps × dt.
  3. Accelerate scalar speed toward max (decelerate near waypoint).
  4. velocity = (cos heading, sin heading, 0) × speed.
  5. Euler integrate position; Z snapped to terrain (VEH-007).

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Template Method + Strategy patterns applied.
NF-CE-005: PRD requirement IDs cited inline.
Implements: VEH-001..007 (road constraint, kinematics, A* pathfinding,
            speed zones stub, EOI flag, signature, terrain lock).
"""
from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.base import Entity, EntityState, EntityType
from sim3dves.entities.context import StepContext
from sim3dves.maps.road_network import RoadNetwork

_D = SimDefaults()


# ### Kinematic parameter bundle ###

@dataclass(frozen=True)
class VehicleKinematics:
    """
    Immutable kinematic parameter set for one vehicle class (VEH-002).

    Attributes
    ----------
    max_speed_mps : float
        Top speed in m/s.
    accel_mps2 : float
        Acceleration rate (m/s²).
    decel_mps2 : float
        Deceleration rate (m/s²), also used for proximity braking.
    max_turn_rate_dps : float
        Maximum yaw rate (degrees / second).
    arrival_threshold_m : float
        Distance at which the vehicle considers a waypoint reached (m).
    """

    max_speed_mps: float = _D.VEH_MAX_SPEED_MPS
    accel_mps2: float = _D.VEH_ACCEL_MPS2
    decel_mps2: float = _D.VEH_DECEL_MPS2
    max_turn_rate_dps: float = _D.VEH_MAX_TURN_RATE_DPS
    arrival_threshold_m: float = _D.VEH_ARRIVAL_THRESHOLD_M


# ### Abstract vehicle base ###

class VehicleEntity(Entity):
    """
    Abstract kinematic vehicle with waypoint path following.

    Concrete _update_behavior / _update_kinematics are provided.
    Subclasses must implement _plan_new_path() to populate self._waypoints
    (Template Method extended with one additional abstract hook).

    Parameters
    ----------
    entity_id : str
        Unique identifier.
    entity_type : EntityType
        WHEELED_VEHICLE or TRACKED_VEHICLE.
    position : np.ndarray
        3-D starting position [x, y, z].  Z overridden to terrain (VEH-007).
    heading : float
        Initial heading in degrees.
    kinematics : VehicleKinematics
        Kinematic envelope parameters.
    is_eoi : bool
        Entity of Interest flag (VEH-005).
    signature : float
        Thermal/optical contrast in [0, 1] (VEH-006).
    rng : np.random.Generator
        Seeded RNG for deterministic destination selection (SIM-003).
    """

    def __init__(
        self,
        entity_id: str,
        entity_type: EntityType,
        position: np.ndarray,
        heading: float,
        kinematics: VehicleKinematics,
        is_eoi: bool,
        signature: float,
        rng: np.random.Generator,
    ) -> None:
        # Terrain lock at construction (VEH-007)
        pos3 = position.astype(float).copy()
        pos3[2] = 0.0

        super().__init__(
            entity_id=entity_id,
            entity_type=entity_type,
            position=pos3,
            velocity=np.zeros(3, dtype=float),
            heading=float(heading),
            is_eoi=is_eoi,
            signature=signature,
        )
        self._kinematics: VehicleKinematics = kinematics
        self._rng: np.random.Generator = rng
        self._speed: float = 0.0           # Current scalar speed (m/s)
        self._waypoints: List[np.ndarray] = []   # 3-D waypoint positions
        self._waypoint_idx: int = 0              # Index of current target

    # ### Template Method hooks (concrete for VehicleEntity) ###

    def _update_behavior(
        self, dt: float, context: Optional[StepContext] = None
    ) -> None:
        """
        Steer toward current waypoint; replan when path is exhausted.

        Called by Entity.step() — do NOT call directly.

        Parameters
        ----------
        dt : float
            Simulation timestep.
        context : StepContext, optional
            Unused in M2; reserved for future inter-vehicle avoidance.
        """
        # Replan if no path or path completed
        if self._waypoint_idx >= len(self._waypoints):
            self._plan_new_path()
            if not self._waypoints:
                # No path available — decelerate to stop and wait
                self._decelerate(dt)
                self.state = EntityState.IDLE
                return

        self._steer_toward_waypoint(dt)
        self.state = EntityState.MOVING

    def _update_kinematics(self, dt: float) -> None:
        """
        Euler-integrate position; enforce terrain lock (VEH-007).

        Z is always set to terrain elevation (flat = 0.0).
        """
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.position[2] = 0.0  # VEH-007: terrain lock (flat terrain)

    @property
    def current_destination(self) -> "np.ndarray | None":
        """
        Return the current target waypoint as [x, y, z], or None (NF-VIZ-011).

        Used by the inspection panel to display where this vehicle is heading.
        Returns None when the path is exhausted or no plan has been computed.

        Returns
        -------
        np.ndarray or None
            3-D waypoint position in metres, or None if no active waypoint.
        """
        if self._waypoint_idx < len(self._waypoints):
            return self._waypoints[self._waypoint_idx]
        return None

    # ### Abstract path-planning hook ###

    @abstractmethod
    def _plan_new_path(self) -> None:
        """
        Select a new destination and populate self._waypoints.

        Must set ``self._waypoints`` to a non-empty list of 3-D positions
        and reset ``self._waypoint_idx = 0``.  If no path can be found,
        ``self._waypoints`` should be left empty.
        """

    # ### Kinematic helpers ###

    def _steer_toward_waypoint(self, dt: float) -> None:
        """
        Kinematic steering step toward ``self._waypoints[self._waypoint_idx]``.

        Steps
        -----
        1. Compute XY vector to target waypoint.
        2. Check arrival radius — advance to next waypoint if reached.
        3. Compute angular error; clamp to max_turn_rate × dt.
        4. Accelerate toward effective max speed; proximity braking.
        5. Update velocity vector from new heading and speed.
        """
        if self._waypoint_idx >= len(self._waypoints):
            return

        target = self._waypoints[self._waypoint_idx]
        diff = target[:2] - self.position[:2]
        dist = float(np.linalg.norm(diff))

        # Arrival check (VEH-002, VEH-003)
        if dist < self._kinematics.arrival_threshold_m:
            self._waypoint_idx += 1
            return

        # Desired heading from ENU vector (radians)
        desired_rad = math.atan2(float(diff[1]), float(diff[0]))
        current_rad = math.radians(self.heading)

        # Shortest angular difference in (-pi, pi]
        delta_rad = desired_rad - current_rad
        delta_rad = (delta_rad + math.pi) % (2.0 * math.pi) - math.pi

        # Clamp yaw to max turn rate (VEH-002)
        max_delta = math.radians(self._kinematics.max_turn_rate_dps) * dt
        delta_rad = float(np.clip(delta_rad, -max_delta, max_delta))

        new_heading_rad = current_rad + delta_rad
        self.heading = math.degrees(new_heading_rad) % 360.0

        # Proximity braking: decelerate when stopping distance >= remaining dist
        stopping_dist = (self._speed ** 2) / (2.0 * self._kinematics.decel_mps2 + 1e-9)
        if dist < stopping_dist:
            self._speed = max(
                0.0, self._speed - self._kinematics.decel_mps2 * dt
            )
        else:
            # Accelerate toward effective max speed
            max_spd = self._effective_max_speed()
            self._speed = min(
                self._speed + self._kinematics.accel_mps2 * dt, max_spd
            )

        # Update velocity vector (VEH-002)
        self.velocity[0] = math.cos(new_heading_rad) * self._speed
        self.velocity[1] = math.sin(new_heading_rad) * self._speed
        self.velocity[2] = 0.0  # Terrain-locked: no vertical velocity

    def _decelerate(self, dt: float) -> None:
        """Reduce speed toward zero when no path is available."""
        self._speed = max(0.0, self._speed - self._kinematics.decel_mps2 * dt)
        heading_rad = math.radians(self.heading)
        self.velocity[0] = math.cos(heading_rad) * self._speed
        self.velocity[1] = math.sin(heading_rad) * self._speed
        self.velocity[2] = 0.0

    def _effective_max_speed(self) -> float:
        """
        Return the max speed applicable in current terrain conditions.

        Base class returns the full ``kinematics.max_speed_mps``.
        TrackedVehicleEntity overrides this for off-road reduction (VEH-001).
        """
        return self._kinematics.max_speed_mps


# ### Concrete: Wheeled Vehicle (road-constrained) ###

class WheeledVehicleEntity(VehicleEntity):
    """
    Wheeled vehicle that follows the road network via A* pathfinding (VEH-001).

    Path-planning strategy: find nearest road node -> A* to random destination.
    On arrival, a new random destination is selected (autonomous patrol).

    Parameters
    ----------
    entity_id : str
        Unique identifier.
    position : np.ndarray
        Initial 3-D position.
    heading : float
        Initial heading (degrees).
    road_network : RoadNetwork
        The world road network used for A* planning (VEH-003).
    kinematics : VehicleKinematics, optional
        Kinematic envelope; defaults to VehicleKinematics().
    is_eoi : bool
        Entity of Interest flag (VEH-005).
    signature : float
        Optical/thermal contrast (VEH-006).
    rng : np.random.Generator, optional
        Seeded RNG for destination selection (SIM-003).
    """

    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        heading: float = 0.0,
        road_network: Optional[RoadNetwork] = None,
        kinematics: Optional[VehicleKinematics] = None,
        is_eoi: bool = False,
        signature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.WHEELED_VEHICLE,
            position=position,
            heading=heading,
            kinematics=kinematics or VehicleKinematics(),
            is_eoi=is_eoi,
            signature=signature,
            rng=rng or np.random.default_rng(),
        )
        # Road network reference required for A* (VEH-003)
        self._road_network: Optional[RoadNetwork] = road_network

    def _plan_new_path(self) -> None:
        """
        A* from nearest road node to a random destination node (VEH-003).

        Sets self._waypoints to the sequence of 3-D node positions along
        the planned route.  Clears waypoints and stays IDLE if the road
        network is unavailable or A* returns no path (disconnected graph).
        """
        if self._road_network is None or len(self._road_network) < 2:
            self._waypoints = []
            return

        all_node_ids = self._road_network.node_ids()

        # Find nearest road node to current position
        from_id = self._road_network.nearest_node(self.position[:2])
        if from_id is None:
            self._waypoints = []
            return

        # Random destination — different from origin
        candidates = [nid for nid in all_node_ids if nid != from_id]
        if not candidates:
            self._waypoints = []
            return

        dest_id = self._rng.choice(candidates)

        # A* pathfinding (VEH-003)
        path_ids = self._road_network.find_path(from_id, dest_id)
        if not path_ids:
            self._waypoints = []  # No path — stay IDLE
            return

        # Convert node IDs to 3-D waypoint positions (Z = terrain = 0)
        self._waypoints = [
            np.array([*self._road_network.node_position(nid), 0.0])
            for nid in path_ids
        ]
        self._waypoint_idx = 0


# ### Concrete: Tracked Vehicle (off-road capable) ###

class TrackedVehicleEntity(VehicleEntity):
    """
    Tracked vehicle that navigates directly to random world positions (VEH-001).

    Unlike wheeled vehicles, tracked vehicles do not require a road network.
    They navigate straight-line toward a randomly selected destination within
    the world boundary.  Speed is reduced by ``off_road_factor`` to represent
    the penalty of cross-country travel (VEH-001, VEH-002).

    Parameters
    ----------
    entity_id : str
        Unique identifier.
    position : np.ndarray
        Initial 3-D position.
    heading : float
        Initial heading (degrees).
    world_extent : np.ndarray
        [X_max, Y_max] world boundary (m) for destination sampling.
    kinematics : VehicleKinematics, optional
        Kinematic envelope; defaults to tracked-vehicle defaults.
    off_road_factor : float
        Speed multiplier when traversing off-road terrain (VEH-001).
    is_eoi : bool
        Entity of Interest flag (VEH-005).
    signature : float
        Optical/thermal contrast (VEH-006).
    rng : np.random.Generator, optional
        Seeded RNG for destination selection (SIM-003).
    """

    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        heading: float = 0.0,
        world_extent: Optional[np.ndarray] = None,
        kinematics: Optional[VehicleKinematics] = None,
        off_road_factor: float = _D.VEH_OFF_ROAD_FACTOR,
        is_eoi: bool = False,
        signature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # Default tracked kinematics have lower top speed
        default_kin = VehicleKinematics(
            max_speed_mps=_D.VEH_TRACKED_MAX_SPEED_MPS,
            accel_mps2=_D.VEH_ACCEL_MPS2,
            decel_mps2=_D.VEH_DECEL_MPS2,
            max_turn_rate_dps=_D.VEH_MAX_TURN_RATE_DPS,
            arrival_threshold_m=_D.VEH_ARRIVAL_THRESHOLD_M,
        )
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.TRACKED_VEHICLE,
            position=position,
            heading=heading,
            kinematics=kinematics or default_kin,
            is_eoi=is_eoi,
            signature=signature,
            rng=rng or np.random.default_rng(),
        )
        # Default to a nominal 1000 × 1000 m world if not provided
        self._world_extent: np.ndarray = (world_extent.astype(float))
        self._off_road_factor: float = float(off_road_factor)

    def _effective_max_speed(self) -> float:
        """
        Off-road speed reduction (VEH-001: tracked MAY traverse off-road
        at reduced speed).
        """
        return self._kinematics.max_speed_mps * self._off_road_factor

    def _plan_new_path(self) -> None:
        """
        Select a random destination within the world boundary (VEH-001).

        Sets self._waypoints to a single 3-D destination point.
        TrackedVehicle navigates straight-line — no road network required.
        """
        dest_xy = self._rng.random(2) * self._world_extent
        # Clamp 10 m inside boundary to avoid immediate OOB kill
        dest_xy = np.clip(dest_xy, 10.0, self._world_extent - 10.0)
        self._waypoints = [np.array([dest_xy[0], dest_xy[1], 0.0])]
        self._waypoint_idx = 0
