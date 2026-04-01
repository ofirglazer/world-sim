"""
sim3dves.entities.uav
=====================
UAVEntity with 3-D kinematic model and full autopilot FSM.

Design Patterns
---------------
* Template Method : UAVEntity inherits Entity.step() skeleton and
  implements _update_behavior / _update_kinematics hooks.
* State (Autopilot FSM) : AutopilotMode enum drives per-step dispatch to
  mode-specific handler methods, replacing raw conditionals.

Autopilot Modes (UAV-005)
--------------------------
WAYPOINT      — Following a search-pattern waypoint list (FLR-008).
LOITER        — Holding orbit around current position (no waypoints).
ORBIT         — Orbiting a payload-cued point at configurable radius/altitude
                (FLR-009); deconfliction assigns PRIMARY/SECONDARY role (FLR-010).
RTB           — Returning to launch position; triggered by low fuel (FLR-006)
                or geofence approach (FLR-005).
EMERGENCY_LAND— Descending to terrain and shutting down.

Flight Rules Implemented
------------------------
FLR-001 NFZ avoidance   : lateral heading deflection when lookahead penetrates NFZ.
FLR-002 Altitude floor  : position[2] clamped ≥ alt_floor_m every step.
FLR-003 Altitude ceiling: position[2] clamped ≤ alt_ceil_m every step.
FLR-004 Separation      : velocity correction when < UAV_SEPARATION_M from peer.
FLR-005 Geofence        : RTB transition when within UAV_GEOFENCE_MARGIN_M of edge.
FLR-006 Low-fuel RTB    : endurance triggers; completes current waypoint then RTB.
FLR-007 Wind model      : configurable constant wind vector applied per step.
FLR-008 Search patterns : LAWNMOWER, EXPANDING_SPIRAL, RANDOM_WALK waypoint lists.
FLR-009 Cued slew       : cue_orbit() public API transitions to ORBIT mode.
FLR-010 Deconfliction   : PRIMARY/SECONDARY role assignment for shared-cue orbits.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Template Method + State patterns applied.
NF-CE-005: PRD requirement IDs cited inline.
Implements: UAV-001..005, FLR-001..010.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.world import NFZCylinder
from sim3dves.entities.base import Entity, EntityState, EntityType
from sim3dves.entities.context import StepContext

_D = SimDefaults()


# ### Enumerations ###

class AutopilotMode(Enum):
    """
    Autopilot FSM state catalogue (UAV-005).

    Values use auto() to satisfy ENT-002 (Enum, not raw strings).
    """
    WAYPOINT = auto()        # Executing search-pattern route (FLR-008)
    LOITER = auto()          # Holding orbit around current position
    ORBIT = auto()           # Cued orbit around payload target (FLR-009)
    RTB = auto()             # Return to base (FLR-005, FLR-006)
    EMERGENCY_LAND = auto()  # Controlled descent and shutdown


class SearchPattern(Enum):
    """Search pattern catalogue (FLR-008)."""
    LAWNMOWER = auto()
    EXPANDING_SPIRAL = auto()
    RANDOM_WALK = auto()


# ### Kinematic parameter bundle ###

@dataclass(frozen=True)
class UAVKinematics:
    """
    Immutable kinematic parameters for a UAV platform (UAV-002).

    Attributes
    ----------
    max_speed_mps : float
        Maximum horizontal speed (m/s).
    climb_rate_mps : float
        Maximum climb rate (m/s).
    descent_rate_mps : float
        Maximum descent rate (m/s).
    turn_rate_dps : float
        Maximum yaw rate (degrees/second).
    arrival_threshold_m : float
        Waypoint arrival radius (m).
    """
    max_speed_mps: float = _D.UAV_MAX_SPEED_MPS
    climb_rate_mps: float = _D.UAV_CLIMB_RATE_MPS
    descent_rate_mps: float = _D.UAV_DESCENT_RATE_MPS
    turn_rate_dps: float = _D.UAV_TURN_RATE_DPS
    arrival_threshold_m: float = _D.UAV_ARRIVAL_THRESHOLD_M


# ### UAV Entity ###

class UAVEntity(Entity):
    """
    Autonomous UAV with 3-D kinematics and a full autopilot FSM (UAV-001..005).

    The UAV operates at a configurable cruise altitude, executes search
    patterns (FLR-008), responds to payload cues (FLR-009), enforces all
    airspace flight rules (FLR-001..010), and models fuel endurance (UAV-003).

    Design Pattern — Template Method
    ---------------------------------
    Inherits Entity.step() which calls _update_behavior() then
    _update_kinematics().  Autopilot mode dispatch is a lightweight State
    pattern: each mode has a dedicated private handler method.

    Parameters
    ----------
    entity_id : str
        Globally unique identifier.
    position : np.ndarray
        Initial 3-D position [x, y, z].  Z is overridden to cruise_altitude_m
        at construction so the UAV starts at the correct altitude (UAV-001).
    heading : float
        Initial heading in degrees.
    launch_position : np.ndarray, optional
        3-D [x, y, z] launch / home point for RTB (FLR-006).  Defaults to
        the initial spawn position.
    kinematics : UAVKinematics, optional
        Kinematic envelope.  Defaults to UAVKinematics() (from SimDefaults).
    search_pattern : SearchPattern
        Initial search pattern to fly (FLR-008).
    cruise_altitude_m : float
        Target operating altitude AGL in metres.
    endurance_s : float
        Total flight endurance budget in seconds (UAV-003).
    world_extent : np.ndarray, optional
        [X_max, Y_max] in metres; used for search-pattern generation and
        geofence checks (FLR-005).  Defaults to (600, 600).
    alt_floor_m : float
        Minimum allowed AGL (FLR-002).
    alt_ceil_m : float
        Maximum allowed AGL (FLR-003).
    nfz_cylinders : list[NFZCylinder], optional
        NFZ volumes for FLR-001 predictive avoidance.
    wind : np.ndarray, optional
        Constant wind vector [vx, vy, vz] in m/s (FLR-007).
        Defaults to (UAV_WIND_X_MPS, UAV_WIND_Y_MPS, UAV_WIND_Z_MPS).
    is_eoi : bool
        Entity of Interest flag.
    signature : float
        Optical/thermal contrast in [0, 1] (VEH-006 schema).
    rng : np.random.Generator, optional
        Seeded RNG for random-walk waypoints and determinism (SIM-003).
    """

    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        heading: float = 0.0,
        launch_position: Optional[np.ndarray] = None,
        kinematics: Optional[UAVKinematics] = None,
        search_pattern: SearchPattern = SearchPattern.LAWNMOWER,
        cruise_altitude_m: float = _D.UAV_CRUISE_ALT_M,
        endurance_s: float = _D.UAV_ENDURANCE_S,
        world_extent: Optional[np.ndarray] = None,
        alt_floor_m: float = _D.UAV_ALT_FLOOR_M,
        alt_ceil_m: float = _D.UAV_ALT_CEIL_M,
        nfz_cylinders: Optional[List[NFZCylinder]] = None,
        wind: Optional[np.ndarray] = None,
        is_eoi: bool = False,
        signature: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # UAV starts at cruise altitude (UAV-001: 3D space)
        pos3 = position.astype(float).copy()
        pos3[2] = float(cruise_altitude_m)

        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.UAV,
            position=pos3,
            velocity=np.zeros(3, dtype=float),
            heading=float(heading),
            is_eoi=is_eoi,
            signature=signature,
        )

        # Kinematic envelope (UAV-002)
        self._kinematics: UAVKinematics = kinematics or UAVKinematics()

        # Endurance budget (UAV-003)
        self.endurance_remaining_s: float = float(endurance_s)
        self._low_fuel: bool = False

        # World spatial context
        self._world_extent: np.ndarray = (
            world_extent.astype(float)
            if world_extent is not None
            else np.array([600.0, 600.0])
        )
        self._alt_floor_m: float = float(alt_floor_m)   # FLR-002
        self._alt_ceil_m: float = float(alt_ceil_m)     # FLR-003
        self._nfz_cylinders: List[NFZCylinder] = nfz_cylinders or []
        self._cruise_altitude_m: float = float(cruise_altitude_m)

        # Wind model (FLR-007)
        if wind is not None:
            self._wind: np.ndarray = wind.astype(float)
        else:
            self._wind = np.array(
                [_D.UAV_WIND_X_MPS, _D.UAV_WIND_Y_MPS, _D.UAV_WIND_Z_MPS],
                dtype=float,
            )

        # Navigation state
        self._search_pattern: SearchPattern = search_pattern
        self._waypoints: List[np.ndarray] = []
        self._waypoint_idx: int = 0

        # Launch / home point for RTB (FLR-006)
        self._launch_position: np.ndarray = (
            launch_position.astype(float)
            if launch_position is not None
            else pos3.copy()
        )

        # Orbit / loiter state (FLR-009)
        self._orbit_center: Optional[np.ndarray] = None
        self._orbit_radius: float = _D.UAV_ORBIT_RADIUS_M
        self._orbit_altitude: float = float(cruise_altitude_m)
        self._orbit_angle: float = float(heading)  # Radians; derived from heading

        # Loiter state (mode=LOITER)
        self._loiter_center: Optional[np.ndarray] = None
        self._loiter_angle: float = 0.0

        # Deconfliction role (FLR-010); determined per-step
        self._deconfliction_role: str = "PRIMARY"

        # Deterministic RNG for random-walk pattern (SIM-003)
        self._rng: np.random.Generator = rng or np.random.default_rng()

        # Initial autopilot mode (UAV-005)
        self._autopilot_mode: AutopilotMode = AutopilotMode.WAYPOINT

    # ### Public properties ###

    @property
    def autopilot_mode(self) -> AutopilotMode:
        """Current autopilot mode (UAV-005). Read-only; use mode methods to change."""
        return self._autopilot_mode

    @property
    def deconfliction_role(self) -> str:
        """PRIMARY or SECONDARY deconfliction role (FLR-010)."""
        return self._deconfliction_role

    @property
    def neighbor_radius_m(self) -> float:
        """
        Larger search radius so UAVs can detect each other for FLR-004/010.

        Returns
        -------
        float
            UAV_NEIGHBOR_RADIUS_M — larger than the ground-entity default.
        """
        return _D.UAV_NEIGHBOR_RADIUS_M

    # ### Public API (cues) ###

    def cue_orbit(
        self,
        center_xy: np.ndarray,
        radius_m: float = _D.UAV_ORBIT_RADIUS_M,
        altitude_m: Optional[float] = None,
    ) -> None:
        """
        Transition to ORBIT mode around *center_xy* (FLR-009).

        Can be called externally by a payload or scenario script to slew
        the UAV onto a persistent orbit around a detected EOI.

        Parameters
        ----------
        center_xy : np.ndarray
            [x, y] orbit centre in metres (ENU).
        radius_m : float
            Orbit radius in metres.
        altitude_m : float, optional
            Orbit altitude AGL.  Defaults to current cruise altitude.
        """
        self._orbit_center = center_xy[:2].astype(float)
        self._orbit_radius = float(radius_m)
        self._orbit_altitude = (
            float(altitude_m) if altitude_m is not None else self._cruise_altitude_m
        )
        # Initialise orbit angle from current position relative to centre
        rel = self.position[:2] - self._orbit_center
        self._orbit_angle = math.atan2(float(rel[1]), float(rel[0]))
        self._autopilot_mode = AutopilotMode.ORBIT

    # ### Template Method hooks ###

    def _update_behavior(
        self, dt: float, context: Optional[StepContext] = None
    ) -> None:
        """
        Execute autopilot FSM for one timestep (UAV-005, FLR-001..010).

        Sequence:
        1. Decrement endurance (UAV-003).
        2. Low-fuel check → RTB if threshold crossed (FLR-006).
        3. Dispatch to mode-specific handler.
        4. Apply flight rules: NFZ avoidance, separation, geofence.
        5. Update deconfliction role (FLR-010).

        Parameters
        ----------
        dt : float
            Simulation timestep.
        context : StepContext, optional
            Neighbour snapshot used for FLR-004 separation and FLR-010.
        """
        # 1. Decrement endurance (UAV-003)
        self.endurance_remaining_s = max(0.0, self.endurance_remaining_s - dt)

        # 2. Low-fuel: trigger RTB once, do not re-trigger (FLR-006)
        if (
            not self._low_fuel
            and self.endurance_remaining_s <= _D.UAV_LOW_FUEL_THRESHOLD_S
        ):
            self._low_fuel = True
            if self._autopilot_mode not in (
                AutopilotMode.RTB, AutopilotMode.EMERGENCY_LAND
            ):
                self._transition_to_rtb()

        # 3. Mode dispatch (State pattern)
        if self._autopilot_mode == AutopilotMode.EMERGENCY_LAND:
            self._do_emergency_land(dt)
        elif self._autopilot_mode == AutopilotMode.RTB:
            self._do_rtb(dt)
        elif self._autopilot_mode == AutopilotMode.ORBIT:
            self._do_orbit(dt)
        elif self._autopilot_mode == AutopilotMode.LOITER:
            self._do_loiter(dt)
        else:
            # WAYPOINT mode
            self._do_waypoint(dt)

        # 4. Flight rules applied after mode handler sets intent velocity
        self._apply_nfz_avoidance()                    # FLR-001
        self._enforce_altitude_velocity()              # FLR-002, FLR-003
        if context is not None:
            self._apply_separation(context)            # FLR-004
        if (
            self._autopilot_mode == AutopilotMode.WAYPOINT
            and self._near_geofence()
        ):
            self._transition_to_rtb()                  # FLR-005

        # 5. Deconfliction role (FLR-010)
        self._update_deconfliction_role(context)

        self.state = EntityState.MOVING

    def _update_kinematics(self, dt: float) -> None:
        """
        Euler-integrate position; apply wind drift and altitude clamping.

        Steps (UAV-001, FLR-002, FLR-003, FLR-007):
        1. Position += (velocity + wind) × dt.
        2. Clamp Z to [alt_floor_m, alt_ceil_m].
        3. Zero vertical velocity if clamp engaged.
        """
        # FLR-007: wind drift added to commanded velocity
        effective_vel = self.velocity + self._wind
        self.position += effective_vel * dt

        # FLR-002 / FLR-003: hard altitude clamping
        if self.position[2] < self._alt_floor_m:
            self.position[2] = self._alt_floor_m
            if self.velocity[2] < 0.0:
                self.velocity[2] = 0.0   # Stop downward velocity at floor
        elif self.position[2] > self._alt_ceil_m:
            self.position[2] = self._alt_ceil_m
            if self.velocity[2] > 0.0:
                self.velocity[2] = 0.0   # Stop upward velocity at ceiling

    # ### Autopilot mode handlers (State pattern) ###

    def _do_waypoint(self, dt: float) -> None:
        """
        Navigate to the next search-pattern waypoint (FLR-008, UAV-005).

        Generates a new waypoint list from the search pattern when the
        current list is exhausted.  Transitions to LOITER if generation
        fails (no world extent configured).
        """
        # (Re)plan if waypoints exhausted
        if self._waypoint_idx >= len(self._waypoints):
            self._waypoints = self._generate_search_waypoints()
            self._waypoint_idx = 0
            if not self._waypoints:
                self._autopilot_mode = AutopilotMode.LOITER
                return

        target = self._waypoints[self._waypoint_idx]
        diff = target - self.position
        dist = float(np.linalg.norm(diff))

        # Arrival check
        if dist < self._kinematics.arrival_threshold_m:
            self._waypoint_idx += 1
            return

        # Compute desired horizontal direction and steer with turn-rate limit
        h_diff = diff[:2]
        h_dist = float(np.linalg.norm(h_diff))

        if h_dist > 1e-6:
            desired_heading_rad = math.atan2(float(h_diff[1]), float(h_diff[0]))
            current_heading_rad = math.radians(self.heading)
            # Shortest angular difference in (-π, π]
            delta_rad = desired_heading_rad - current_heading_rad
            delta_rad = (delta_rad + math.pi) % (2.0 * math.pi) - math.pi
            max_delta = math.radians(self._kinematics.turn_rate_dps) * dt
            delta_rad = float(np.clip(delta_rad, -max_delta, max_delta))
            new_heading_rad = current_heading_rad + delta_rad
            self.heading = math.degrees(new_heading_rad) % 360.0

            # Horizontal speed: full speed unless very close (proximity brake)
            h_speed = min(self._kinematics.max_speed_mps, h_dist / dt)
            h_speed = min(h_speed, self._kinematics.max_speed_mps)
            self.velocity[0] = math.cos(new_heading_rad) * h_speed
            self.velocity[1] = math.sin(new_heading_rad) * h_speed

        # Vertical component: climb/descend toward target altitude
        v_diff = diff[2]
        self.velocity[2] = float(np.clip(
            v_diff / max(dt, 0.1),
            -self._kinematics.descent_rate_mps,
            self._kinematics.climb_rate_mps,
        ))

    def _do_orbit(self, dt: float) -> None:
        """
        Fly a circular orbit around ``_orbit_center`` (FLR-009, FLR-010).

        Tangential velocity provides smooth circular motion without
        steering lag.  Altitude control climbs/descends toward orbit altitude.

        SECONDARY UAVs (FLR-010) orbit at an altitude offset above PRIMARY.
        """
        if self._orbit_center is None:
            # No cue defined — fall back to LOITER
            self._autopilot_mode = AutopilotMode.LOITER
            return

        # Advance orbit angle: ω = v / r  (radians per second)
        angular_vel = _D.UAV_ORBIT_SPEED_MPS / max(self._orbit_radius, 1.0)
        self._orbit_angle += angular_vel * dt

        # Tangential velocity: perpendicular to radius vector, CCW positive
        self.velocity[0] = -math.sin(self._orbit_angle) * _D.UAV_ORBIT_SPEED_MPS
        self.velocity[1] = math.cos(self._orbit_angle) * _D.UAV_ORBIT_SPEED_MPS

        # FLR-010: altitude offset for SECONDARY role
        target_alt = self._orbit_altitude
        if self._deconfliction_role == "SECONDARY":
            target_alt += _D.UAV_DECONFLICTION_ALT_STEP_M

        # Altitude control toward orbit altitude
        alt_error = target_alt - self.position[2]
        self.velocity[2] = float(np.clip(
            alt_error / max(dt, 0.1),
            -self._kinematics.descent_rate_mps,
            self._kinematics.climb_rate_mps,
        ))

    def _do_loiter(self, dt: float) -> None:
        """
        Hold a small circular orbit around the loiter centre.

        The loiter centre is captured on first entry from the current
        position.  Uses a reduced speed and smaller radius than full orbit.
        """
        # Capture loiter centre on mode entry
        if self._loiter_center is None:
            self._loiter_center = self.position[:2].copy()
            rel = self.position[:2] - self._loiter_center
            self._loiter_angle = math.atan2(float(rel[1]), float(rel[0]))

        # Advance loiter angle
        angular_vel = _D.UAV_LOITER_SPEED_MPS / max(_D.UAV_LOITER_RADIUS_M, 1.0)
        self._loiter_angle += angular_vel * dt

        # Tangential velocity (reduced speed for holding pattern)
        self.velocity[0] = -math.sin(self._loiter_angle) * _D.UAV_LOITER_SPEED_MPS
        self.velocity[1] = math.cos(self._loiter_angle) * _D.UAV_LOITER_SPEED_MPS

        # Altitude: hold cruise altitude
        alt_error = self._cruise_altitude_m - self.position[2]
        self.velocity[2] = float(np.clip(
            alt_error / max(dt, 0.1),
            -self._kinematics.descent_rate_mps,
            self._kinematics.climb_rate_mps,
        ))

    def _do_rtb(self, dt: float) -> None:
        """
        Fly directly to launch position (FLR-005, FLR-006).

        Transitions to EMERGENCY_LAND on arrival at the launch point.
        Uses full horizontal speed without proximity braking to ensure
        the UAV reaches home before fuel runs out.
        """
        diff = self._launch_position - self.position
        h_diff = diff[:2]
        h_dist = float(np.linalg.norm(h_diff))

        # Arrival at launch point
        if h_dist < self._kinematics.arrival_threshold_m:
            self._autopilot_mode = AutopilotMode.EMERGENCY_LAND
            return

        # Steer toward launch position at full horizontal speed
        if h_dist > 1e-6:
            h_dir = h_diff / h_dist
            # Apply turn-rate limit for physical realism (UAV-002)
            desired_heading_rad = math.atan2(float(h_dir[1]), float(h_dir[0]))
            current_heading_rad = math.radians(self.heading)
            delta_rad = desired_heading_rad - current_heading_rad
            delta_rad = (delta_rad + math.pi) % (2.0 * math.pi) - math.pi
            max_delta = math.radians(self._kinematics.turn_rate_dps) * dt
            delta_rad = float(np.clip(delta_rad, -max_delta, max_delta))
            new_heading_rad = current_heading_rad + delta_rad
            self.heading = math.degrees(new_heading_rad) % 360.0

            self.velocity[0] = math.cos(new_heading_rad) * self._kinematics.max_speed_mps
            self.velocity[1] = math.sin(new_heading_rad) * self._kinematics.max_speed_mps

        # Descend toward launch altitude during RTB
        v_diff = self._launch_position[2] - self.position[2]
        self.velocity[2] = float(np.clip(
            v_diff / max(dt, 0.1),
            -self._kinematics.descent_rate_mps,
            self._kinematics.climb_rate_mps,
        ))

    def _do_emergency_land(self, dt: float) -> None:
        """
        Controlled descent to terrain; kills entity on touchdown.

        Horizontal motion is zeroed; the UAV descends at max descent rate.
        Entity is killed when Z drops to within 1 m of the altitude floor.
        """
        self.velocity[0] = 0.0
        self.velocity[1] = 0.0
        self.velocity[2] = -self._kinematics.descent_rate_mps

        # Kill when close to ground (altitude floor ~= terrain)
        if self.position[2] <= self._alt_floor_m + 1.0:
            self.kill()

    # ### Flight rule helpers ###

    def _apply_nfz_avoidance(self) -> None:
        """
        Lateral velocity deflection when lookahead penetrates any NFZ (FLR-001).

        The UAV predicts its position UAV_NFZ_LOOKAHEAD_S seconds ahead.
        If the lookahead point is inside an NFZ, horizontal velocity is
        rotated 90° away from the NFZ centre, preserving speed magnitude.
        """
        lookahead = self.position + self.velocity * _D.UAV_NFZ_LOOKAHEAD_S
        for nfz in self._nfz_cylinders:
            if nfz.contains(lookahead):
                # Direction from NFZ centre to UAV (pointing outward)
                to_uav = self.position[:2] - nfz.center_xy
                dist = float(np.linalg.norm(to_uav))
                if dist < 1e-6:
                    dist = 1.0
                    to_uav = np.array([1.0, 0.0])
                outward = to_uav / dist

                # Perpendicular (tangential to NFZ edge) deflection
                perp = np.array([-outward[1], outward[0]])
                h_speed = float(np.linalg.norm(self.velocity[:2]))
                if h_speed < 1.0:
                    h_speed = self._kinematics.max_speed_mps * 0.5

                # Weight: 50% outward + 50% perpendicular for smooth avoidance
                avoid = 0.5 * outward + 0.5 * perp
                avoid /= float(np.linalg.norm(avoid) + 1e-9)
                self.velocity[0] = avoid[0] * h_speed
                self.velocity[1] = avoid[1] * h_speed
                break  # One NFZ avoidance per step is sufficient

    def _enforce_altitude_velocity(self) -> None:
        """
        Clamp vertical velocity to climb/descent rate limits (UAV-002).

        Applied after mode handlers so neither the FSM nor flight-rule
        corrections can exceed the kinematic envelope.
        """
        self.velocity[2] = float(np.clip(
            self.velocity[2],
            -self._kinematics.descent_rate_mps,
            self._kinematics.climb_rate_mps,
        ))

    def _apply_separation(self, context: StepContext) -> None:
        """
        Velocity correction to maintain ≥ UAV_SEPARATION_M from peer UAVs (FLR-004).

        A repulsion force proportional to proximity is added to horizontal
        velocity, then clamped back to max_speed_mps.

        Parameters
        ----------
        context : StepContext
            Neighbour snapshot; only UAV entities are considered.
        """
        for neighbor in context.neighbors:
            if neighbor.entity_type != EntityType.UAV:
                continue   # Only UAV-UAV separation (FLR-004)
            diff = self.position - neighbor.position
            dist3d = float(np.linalg.norm(diff))
            if 0.0 < dist3d < _D.UAV_SEPARATION_M:
                # Repulsion strength: scales from max at 0 to 0 at threshold
                strength = (
                    (1.0 - dist3d / _D.UAV_SEPARATION_M)
                    * self._kinematics.max_speed_mps
                )
                avoidance_dir = diff / dist3d
                self.velocity[0] += avoidance_dir[0] * strength
                self.velocity[1] += avoidance_dir[1] * strength

                # Clamp horizontal speed to kinematic limit
                h_speed = float(np.linalg.norm(self.velocity[:2]))
                if h_speed > self._kinematics.max_speed_mps:
                    scale = self._kinematics.max_speed_mps / h_speed
                    self.velocity[0] *= scale
                    self.velocity[1] *= scale

    def _near_geofence(self) -> bool:
        """
        Return True if the UAV is within UAV_GEOFENCE_MARGIN_M of any world edge.

        Used to trigger RTB before boundary exit (FLR-005).

        Returns
        -------
        bool
            True if geofence approach threshold is breached.
        """
        m = _D.UAV_GEOFENCE_MARGIN_M
        return (
            self.position[0] < m
            or self.position[0] > self._world_extent[0] - m
            or self.position[1] < m
            or self.position[1] > self._world_extent[1] - m
        )

    def _transition_to_rtb(self) -> None:
        """Cleanly transition to RTB mode (FLR-005, FLR-006)."""
        self._autopilot_mode = AutopilotMode.RTB
        self._waypoints = []
        self._waypoint_idx = 0
        # Reset loiter centre so LOITER can re-capture if needed later
        self._loiter_center = None

    def _update_deconfliction_role(
        self, context: Optional[StepContext]
    ) -> None:
        """
        Assign PRIMARY or SECONDARY deconfliction role (FLR-010).

        When two or more UAVs are in ORBIT mode with overlapping cue centres
        (within UAV_SAME_ORBIT_THRESHOLD_M), the UAV with the lexicographically
        smallest entity_id gets PRIMARY; all others get SECONDARY.

        Parameters
        ----------
        context : StepContext, optional
            Neighbour snapshot; UAV peers examined for ORBIT mode.
        """
        # Only relevant in ORBIT mode
        if self._autopilot_mode != AutopilotMode.ORBIT or context is None:
            self._deconfliction_role = "PRIMARY"
            return

        threshold = _D.UAV_SAME_ORBIT_THRESHOLD_M

        for neighbor in context.neighbors:
            if neighbor.entity_type != EntityType.UAV:
                continue
            # Compare autopilot modes via public property (no private access)
            if not isinstance(neighbor, UAVEntity):
                continue
            if neighbor.autopilot_mode != AutopilotMode.ORBIT:
                continue
            # Skip if neighbor has no valid orbit centre
            if neighbor._orbit_center is None or self._orbit_center is None:
                continue
            # Check if orbiting the same cue point
            centre_dist = float(
                np.linalg.norm(neighbor._orbit_center - self._orbit_center)
            )
            if centre_dist > threshold:
                continue
            # A peer UAV is orbiting the same point — role by entity_id order
            if neighbor.entity_id < self.entity_id:
                self._deconfliction_role = "SECONDARY"
                return

        self._deconfliction_role = "PRIMARY"

    # ### Search pattern generators (FLR-008) ###

    def _generate_search_waypoints(self) -> List[np.ndarray]:
        """
        Generate a waypoint list from the configured search pattern (FLR-008).

        Returns
        -------
        list[np.ndarray]
            3-D waypoints at cruise altitude.  Empty list if world extent
            is not configured.
        """
        if self._search_pattern == SearchPattern.LAWNMOWER:
            return self._generate_lawnmower()
        elif self._search_pattern == SearchPattern.EXPANDING_SPIRAL:
            return self._generate_expanding_spiral()
        else:
            # RANDOM_WALK
            return self._generate_random_walk()

    def _generate_lawnmower(self) -> List[np.ndarray]:
        """
        Parallel-strip lawnmower covering the world extent (FLR-008).

        Strips run West→East and East→West alternately to minimise turns.
        """
        strip_w = _D.UAV_LAWNMOWER_STRIP_W_M
        margin = _D.UAV_GEOFENCE_MARGIN_M
        x_min = margin
        x_max = self._world_extent[0] - margin
        y_min = margin
        y_max = self._world_extent[1] - margin
        z = self._cruise_altitude_m

        waypoints: List[np.ndarray] = []
        y = y_min
        strip_idx = 0
        while y <= y_max + strip_w * 0.5:  # +0.5 * strip_w avoids floating-point skip
            # Alternate direction each strip (boustrophedon / lawnmower)
            if strip_idx % 2 == 0:
                waypoints.append(np.array([x_min, y, z]))
                waypoints.append(np.array([x_max, y, z]))
            else:
                waypoints.append(np.array([x_max, y, z]))
                waypoints.append(np.array([x_min, y, z]))
            y += strip_w
            strip_idx += 1

        return waypoints

    def _generate_expanding_spiral(self) -> List[np.ndarray]:
        """
        Outward-expanding spiral from world centre (FLR-008).

        Each ring adds UAV_SPIRAL_STRIP_W_M to the radius with 8+ points
        per ring so the angular step is ≤ strip_width / radius.
        """
        strip_w = _D.UAV_SPIRAL_STRIP_W_M
        margin = _D.UAV_GEOFENCE_MARGIN_M
        cx = self._world_extent[0] / 2.0
        cy = self._world_extent[1] / 2.0
        max_radius = min(cx, cy) - margin
        z = self._cruise_altitude_m

        waypoints: List[np.ndarray] = []
        radius = strip_w
        base_angle = 0.0
        while radius <= max_radius:
            # More points for larger radii to maintain angular resolution
            n_points = max(8, int(2.0 * math.pi * radius / strip_w))
            for i in range(n_points):
                angle = base_angle + i * 2.0 * math.pi / n_points
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                waypoints.append(np.array([x, y, z]))
            radius += strip_w
            # Slight rotation per ring for denser coverage
            base_angle += math.pi / 4.0

        return waypoints

    def _generate_random_walk(self) -> List[np.ndarray]:
        """
        Random sequence of UAV_RANDOM_WALK_WAYPOINTS positions (FLR-008).

        Each point is sampled uniformly within the world minus the geofence
        margin.  Uses the entity's seeded RNG (SIM-003).
        """
        n = _D.UAV_RANDOM_WALK_WAYPOINTS
        margin = _D.UAV_GEOFENCE_MARGIN_M
        z = self._cruise_altitude_m

        waypoints: List[np.ndarray] = []
        for _ in range(n):
            x = float(self._rng.uniform(margin, self._world_extent[0] - margin))
            y = float(self._rng.uniform(margin, self._world_extent[1] - margin))
            waypoints.append(np.array([x, y, z]))

        return waypoints
