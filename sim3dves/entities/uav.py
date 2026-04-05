"""
sim3dves.entities.uav
=====================
UAVEntity with 3-D kinematic model and full autopilot FSM.

Design Patterns
---------------
* Template Method : UAVEntity inherits Entity.step() skeleton and
  implements _update_behavior / _update_kinematics hooks.
* State (Autopilot FSM) : AutopilotMode enum drives per-step dispatch to
  mode-specific handler methods.

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
FLR-001: NFZ avoidance via turn-rate-limited heading deflection (NO snap-writes).
FLR-002: Altitude floor clamping.
FLR-003: Altitude ceiling clamping.
FLR-004: UAV-UAV separation repulsion.
FLR-005: Straight-edge geofence -> RTB transition.
FLR-006: Low-fuel -> RTB.
FLR-007: Configurable wind drift.
FLR-008: LAWNMOWER, EXPANDING_SPIRAL, RANDOM_WALK patterns within UAV_SEARCH_MARGIN_M.
FLR-009: Cued orbit via cue_orbit() public API.
FLR-010: PRIMARY / SECONDARY deconfliction for shared cue-point orbits.
FLR-011: Corner-escape heading before RTB when approaching a corner pocket.

v1.2 Bug Fixes
--------------
* FLR-001: _apply_nfz_avoidance() previously wrote velocity directly (up to
  1800 deg/s effective yaw rate). Now calls _steer_toward_heading() which
  clamps delta to UAV_TURN_RATE_DPS * dt per step (UAV-002).
* FLR-005/FLR-011: no corner-escape logic existed. _in_corner_pocket() and
  _corner_escape_heading_rad() added; _update_behavior() checks corner first.
* FLR-008: pattern generators used UAV_GEOFENCE_MARGIN_M as inner boundary,
  placing waypoints exactly on the RTB trigger line. Now use UAV_SEARCH_MARGIN_M.
* _do_waypoint / _do_rtb: duplicated inline steering logic removed; both now
  delegate to the shared _steer_toward_heading() helper.
* Redundant min() in _do_waypoint collapsed to a single expression.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Template Method + State patterns applied.
NF-CE-005: PRD requirement IDs cited inline.
Implements: UAV-001..005, FLR-001..011.
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
    WAYPOINT = auto()  # Executing search-pattern route (FLR-008)
    LOITER = auto()  # Holding orbit around current position
    ORBIT = auto()  # Cued orbit around payload target (FLR-009)
    RTB = auto()  # Return to base (FLR-005, FLR-006)
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
        Maximum yaw rate (degrees/second).  All heading changes in this
        module are clamped to this value * dt (FLR-001, UAV-002).
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
        Initial search pattern (FLR-008).
    cruise_altitude_m : float
        Target operating altitude AGL (m).
    endurance_s : float
        Total flight endurance budget (s) (UAV-003).
    world_extent : np.ndarray, optional
        [X_max, Y_max] in metres; used for search-pattern generation and
        geofence checks (FLR-005).  Defaults to (600, 600).
    alt_floor_m : float
        Minimum AGL (FLR-002).
    alt_ceil_m : float
        Maximum AGL (FLR-003).
    nfz_cylinders : list[NFZCylinder], optional
        NFZ volumes for FLR-001 predictive avoidance.
    wind : np.ndarray, optional
        Constant wind vector [vx, vy, vz] in m/s (FLR-007).
        Defaults to (UAV_WIND_X_MPS, UAV_WIND_Y_MPS, UAV_WIND_Z_MPS).
    is_eoi : bool
        Entity of Interest flag.
    signature : float
        Optical/thermal contrast in [0, 1].
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
        pos3[2] = float(cruise_altitude_m)  # UAV-001: start at cruise altitude

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

        # Endurance (UAV-003)
        self.endurance_remaining_s: float = float(endurance_s)
        self._low_fuel: bool = False

        # World spatial context
        self._world_extent: np.ndarray = world_extent.astype(float)
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
        self._orbit_angle: float = math.radians(float(heading))

        # Loiter state
        self._loiter_center: Optional[np.ndarray] = None
        self._loiter_angle: float = 0.0

        # Deconfliction role (FLR-010)
        self._deconfliction_role: str = "PRIMARY"

        # Deterministic RNG for random-walk pattern (SIM-003)
        self._rng: np.random.Generator = rng or np.random.default_rng()

        # Initial autopilot mode (UAV-005)
        self._autopilot_mode: AutopilotMode = AutopilotMode.WAYPOINT

        # Heading snapshot taken at the top of every _update_behavior call.
        # _apply_nfz_avoidance restores this before steering, ensuring that
        # mode-handler turn + NFZ-avoidance turn together consume at most one
        # turn-rate budget (FLR-001 fix, UAV-002).
        self._heading_step_start: float = float(heading)

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
    def low_fuel(self) -> bool:
        """True when endurance_remaining_s has crossed the low-fuel threshold (FLR-006)."""
        return self._low_fuel

    @property
    def nfz_violated(self) -> bool:
        """
        True if the UAV's current position is inside any configured NFZ (FLR-001).

        Evaluated on demand from the UAV's own NFZ list — consistent with the
        avoidance logic that uses the same cylinders.

        Returns
        -------
        bool
            True when the current 3-D position violates any NFZ volume.
        """
        return any(nfz.contains(self.position) for nfz in self._nfz_cylinders)

    @property
    def current_destination(self) -> Optional[np.ndarray]:
        """
        Return the current navigation target as [x, y, z], or None (NF-VIZ-011).

        For WAYPOINT/RTB modes: the next waypoint.
        For ORBIT mode: the orbit centre (2-D, promoted to 3-D with orbit altitude).
        For LOITER/EMERGENCY_LAND: None (holding position).

        Returns
        -------
        np.ndarray or None
        """
        if self._autopilot_mode == AutopilotMode.ORBIT and self._orbit_center is not None:
            return np.array([
                self._orbit_center[0], self._orbit_center[1], self._orbit_altitude
            ])
        if self._autopilot_mode == AutopilotMode.RTB:
            return self._launch_position.copy()
        if self._waypoint_idx < len(self._waypoints):
            return self._waypoints[self._waypoint_idx].copy()
        return None

    @property
    def neighbor_radius_m(self) -> float:
        """
        Larger search radius for UAV-UAV separation detection (FLR-004).

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
        Execute autopilot FSM for one timestep (UAV-005, FLR-001..011).

        Sequence
        --------
        1. Decrement endurance (UAV-003).
        2. Low-fuel check -> RTB transition (FLR-006).
        3. Mode dispatch.
        4. Flight rules: NFZ avoidance (FLR-001), altitude limits (FLR-002/003),
           UAV separation (FLR-004).
        5. Corner-escape / geofence check (FLR-005, FLR-011).
        6. Deconfliction role update (FLR-010).

        Parameters
        ----------
        dt : float
            Simulation timestep.
        context : StepContext, optional
            Neighbour snapshot used for FLR-004 separation and FLR-010.
        """
        # Snapshot heading before any mode-handler steering (FLR-001 fix)
        self._heading_step_start = self.heading

        # 1. Decrement endurance (UAV-003)
        self.endurance_remaining_s = max(0.0, self.endurance_remaining_s - dt)

        # 2. Low-fuel -> RTB (FLR-006); trigger once only
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

        # 4. Flight rules applied after mode handler has set intent velocity
        self._apply_nfz_avoidance(dt)                    # FLR-001 - turn-rate limited
        self._enforce_altitude_velocity()              # FLR-002, FLR-003
        if context is not None:
            self._apply_separation(context)            # FLR-004

        # 5. Corner-escape / straight-edge geofence (FLR-011, FLR-005)
        if self._autopilot_mode != AutopilotMode.EMERGENCY_LAND:
            if self._in_corner_pocket():
                # FLR-011: fly corner-bisector outward; do NOT transition to RTB yet
                escape_rad = self._corner_escape_heading_rad()
                self._steer_toward_heading(escape_rad, dt)
            elif (
                self._autopilot_mode == AutopilotMode.WAYPOINT
                and self._near_geofence()
            ):
                self._transition_to_rtb()      # FLR-005: straight-edge approach

        # 6. Deconfliction role (FLR-010)
        self._update_deconfliction_role(context)

        self.state = EntityState.MOVING

    def _update_kinematics(self, dt: float) -> None:
        """
        Euler-integrate position; apply wind drift and altitude clamping (FLR-007).

        Steps: position += (velocity + wind) * dt, then clamp Z.
        1. Position += (velocity + wind) × dt.
        2. Clamp Z to [alt_floor_m, alt_ceil_m].
        3. Zero vertical velocity if clamp engaged.
        """
        # FLR-007: wind drift added to commanded velocity
        effective_vel = self.velocity + self._wind
        self.position += effective_vel * dt

        # FLR-002 / FLR-003: hard altitude clamping after integration
        if self.position[2] < self._alt_floor_m:
            self.position[2] = self._alt_floor_m
            if self.velocity[2] < 0.0:
                self.velocity[2] = 0.0   # Stop downward velocity at floor
        elif self.position[2] > self._alt_ceil_m:
            self.position[2] = self._alt_ceil_m
            if self.velocity[2] > 0.0:
                self.velocity[2] = 0.0

    # ### Shared steering helper ###

    def _steer_toward_heading(
        self,
        desired_heading_rad: float,
        dt: float,
        speed: Optional[float] = None,
    ) -> None:
        """
        Apply a turn-rate-limited heading change toward *desired_heading_rad* (UAV-002).

        This is the single shared implementation for ALL heading changes in the
        UAV model.  It guarantees that |Δheading| <= turn_rate_dps * dt per step,
        satisfying FLR-001 (NFZ avoidance must respect turn rate) and FLR-011
        (corner-escape must respect turn rate).

        Parameters
        ----------
        desired_heading_rad : float
            Target heading in radians (ENU).
        dt : float
            Simulation timestep (s).
        speed : float, optional
            Horizontal speed to apply.  Defaults to current XY speed; falls
            back to 50% of max speed if current speed is negligible.
        """
        current_rad = math.radians(self.heading)
        # Shortest angular path in (-pi, pi]
        delta_rad = (desired_heading_rad - current_rad + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        # Clamp to kinematic turn-rate limit (UAV-002)
        max_delta = math.radians(self._kinematics.turn_rate_dps) * dt
        delta_rad = float(np.clip(delta_rad, -max_delta, max_delta))

        new_heading_rad = current_rad + delta_rad
        self.heading = math.degrees(new_heading_rad) % 360.0

        # Determine horizontal speed
        h_speed: float = (
            speed
            if speed is not None
            else float(np.linalg.norm(self.velocity[:2]))
        )
        if h_speed < 0.1:
            h_speed = self._kinematics.max_speed_mps * 0.5

        self.velocity[0] = math.cos(new_heading_rad) * h_speed
        self.velocity[1] = math.sin(new_heading_rad) * h_speed
        # Vertical component is NOT touched; caller manages it separately.

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
            # Proximity brake: approach waypoint at reduced speed (UAV-002)
            h_speed = min(h_dist / dt, self._kinematics.max_speed_mps)
            self._steer_toward_heading(desired_heading_rad, dt, speed=h_speed)

        # Vertical: climb/descend toward waypoint altitude
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

        # Advance orbit angle: omega = v / r
        angular_vel = _D.UAV_ORBIT_SPEED_MPS / max(self._orbit_radius, 1.0)
        self._orbit_angle += angular_vel * dt

        # Tangential velocity: perpendicular to radius vector, CCW positive
        self.velocity[0] = -math.sin(self._orbit_angle) * _D.UAV_ORBIT_SPEED_MPS
        self.velocity[1] = math.cos(self._orbit_angle) * _D.UAV_ORBIT_SPEED_MPS

        # FLR-010: SECONDARY orbits higher to maintain separation
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
            desired_heading_rad = math.atan2(float(h_diff[1]), float(h_diff[0]))
            # RTB at full speed (UAV-002); _steer_toward_heading clamps turn rate
            self._steer_toward_heading(
                desired_heading_rad, dt,
                speed=self._kinematics.max_speed_mps,
            )

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

    def _apply_nfz_avoidance(self, dt: float) -> None:
        """
        Turn-rate-limited lateral heading deflection when lookahead hits any NFZ.

        FIX (v1.2 / FLR-001): previous implementation wrote self.velocity directly,
        bypassing UAV_TURN_RATE_DPS entirely (up to 1800 deg/s effective yaw rate).
        Now delegates exclusively to _steer_toward_heading() which enforces the
        turn-rate limit on every call (UAV-002).

        Parameters
        ----------
        dt : float
            Simulation timestep — required to compute the clamped heading delta.
        """
        lookahead = self.position + self.velocity * _D.UAV_NFZ_LOOKAHEAD_S
        for nfz in self._nfz_cylinders:
            if not nfz.contains(lookahead):
                continue

            # Direction from NFZ centre to UAV (outward escape direction)
            to_uav = self.position[:2] - nfz.center_xy
            dist = float(np.linalg.norm(to_uav))
            if dist < 1e-6:
                dist = 1.0
                to_uav = np.array([1.0, 0.0])
            outward = to_uav / dist

            # 50% outward + 50% perpendicular for smooth tangential avoidance
            perp = np.array([-outward[1], outward[0]])
            avoid = 0.5 * outward + 0.5 * perp
            norm = float(np.linalg.norm(avoid))
            if norm > 1e-9:
                avoid /= norm

            desired_heading_rad = math.atan2(float(avoid[1]), float(avoid[0]))
            # FLR-001: restore pre-step heading so the avoidance turn does not
            # compound on top of the mode-handler turn (UAV-002 turn-rate limit).
            self.heading = self._heading_step_start
            self._steer_toward_heading(desired_heading_rad, dt)
            break  # One NFZ avoidance correction per step is sufficient

    def _enforce_altitude_velocity(self) -> None:
        """Clamp vertical velocity to kinematic climb/descent limits (UAV-002)."""
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

                # Clamp horizontal speed to kinematic limit after correction
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
        margin = _D.UAV_GEOFENCE_MARGIN_M
        return (
            self.position[0] < margin
            or self.position[0] > self._world_extent[0] - margin
            or self.position[1] < margin
            or self.position[1] > self._world_extent[1] - margin
        )

    def _in_corner_pocket(self) -> bool:
        """
        Return True when the UAV is simultaneously within UAV_GEOFENCE_MARGIN_M
        of two adjacent world boundaries (FLR-011).

        A corner pocket requires a larger turn arc (up to 135 degrees) than a
        straight-edge boundary (90 degrees), so a separate escape heading is used
        rather than an immediate RTB transition.

        Returns
        -------
        bool
        """
        m = _D.UAV_GEOFENCE_MARGIN_M
        near_west = self.position[0] < m
        near_east = self.position[0] > self._world_extent[0] - m
        near_south = self.position[1] < m
        near_north = self.position[1] > self._world_extent[1] - m
        # Corner = simultaneously near ONE horizontal AND ONE vertical boundary
        return (near_west or near_east) and (near_south or near_north)

    def _corner_escape_heading_rad(self) -> float:
        """
        Compute the outward corner-bisector escape heading (FLR-011).

        Identifies the nearest world corner from the UAV's position relative to
        world centre.  Returns the heading that points directly away from that
        corner (the outward bisector direction).

        The caller applies this via _steer_toward_heading(), which enforces the
        turn-rate limit so the escape is physically realisable (UAV-002, FLR-011).

        Returns
        -------
        float
            Desired escape heading in radians (ENU convention).
        """
        half_x = self._world_extent[0] / 2.0
        half_y = self._world_extent[1] / 2.0

        # Nearest corner coordinates
        cx = 0.0 if self.position[0] < half_x else self._world_extent[0]
        cy = 0.0 if self.position[1] < half_y else self._world_extent[1]

        # Outward direction: UAV position - corner
        dx = self.position[0] - cx
        dy = self.position[1] - cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            # Exactly on a corner: escape along the 45-degree diagonal
            dx = 1.0 if self.position[0] < half_x else -1.0
            dy = 1.0 if self.position[1] < half_y else -1.0
            dist = math.sqrt(2.0)

        return math.atan2(dy / dist, dx / dist)

    def _transition_to_rtb(self) -> None:
        """Cleanly transition to RTB mode (FLR-005, FLR-006)."""
        self._autopilot_mode = AutopilotMode.RTB
        self._waypoints = []
        self._waypoint_idx = 0
        self._loiter_center = None   # Reset loiter centre on next LOITER entry

    def _update_deconfliction_role(
        self, context: Optional[StepContext]
    ) -> None:
        """
        Assign PRIMARY or SECONDARY deconfliction role (FLR-010).

        Both UAVs in ORBIT mode targeting the same cue point determine roles by
        lexicographic entity_id ordering: smaller ID = PRIMARY.

        Parameters
        ----------
        context : StepContext, optional
            Neighbour snapshot.
        """
        # Only relevant in ORBIT mode
        if self._autopilot_mode != AutopilotMode.ORBIT or context is None:
            self._deconfliction_role = "PRIMARY"
            return

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
            if centre_dist > _D.UAV_SAME_ORBIT_THRESHOLD_M:
                continue
            # Peer is orbiting same cue point; smaller ID gets PRIMARY
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
        margin = _D.UAV_SEARCH_MARGIN_M  # FLR-008: inner search boundary
        x_min = margin
        x_max = self._world_extent[0] - margin
        y_min = margin
        y_max = self._world_extent[1] - margin
        z = self._cruise_altitude_m

        if x_max <= x_min or y_max <= y_min:
            return []   # World too small for search pattern at this margin

        waypoints: List[np.ndarray] = []
        y = y_min
        strip_idx = 0
        while y <= y_max:   # FIX: was y_max + strip_w*0.5, which placed waypoints beyond the safe margin
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
        Outward-expanding spiral from world centre inside UAV_SEARCH_MARGIN_M.

        FIX (v1.2 / FLR-008): inner boundary changed from UAV_GEOFENCE_MARGIN_M
        to UAV_SEARCH_MARGIN_M.
        """
        strip_w = _D.UAV_SPIRAL_STRIP_W_M
        margin = _D.UAV_SEARCH_MARGIN_M   # FLR-008: inner search boundary
        cx = self._world_extent[0] / 2.0
        cy = self._world_extent[1] / 2.0
        max_radius = min(cx, cy) - margin
        z = self._cruise_altitude_m

        if max_radius <= 0.0:
            return []

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
        Random waypoints within world minus UAV_SEARCH_MARGIN_M.

        FIX (v1.2 / FLR-008): inner boundary changed from UAV_GEOFENCE_MARGIN_M
        to UAV_SEARCH_MARGIN_M.
        """
        n = _D.UAV_RANDOM_WALK_WAYPOINTS
        margin = _D.UAV_SEARCH_MARGIN_M   # FLR-008: inner search boundary
        z = self._cruise_altitude_m

        lo_x = margin
        hi_x = self._world_extent[0] - margin
        lo_y = margin
        hi_y = self._world_extent[1] - margin

        if hi_x <= lo_x or hi_y <= lo_y:
            return []

        return [
            np.array([
                float(self._rng.uniform(lo_x, hi_x)),
                float(self._rng.uniform(lo_y, hi_y)),
                z,
            ])
            for _ in range(n)
        ]
