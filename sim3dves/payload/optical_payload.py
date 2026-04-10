"""
sim3dves.payload.optical_payload
=================================
OpticalPayload: EO/IR sensor model attached to a UAVEntity.

Design Patterns
---------------
* State (Gimbal FSM) : GimbalMode enum drives per-step dispatch.
* Facade             : OpticalPayload exposes a single step() call that
                       orchestrates gimbal articulation, FOV projection,
                       LOS batch raycast, and detection; internal complexity
                       is hidden from UAVEntity and SimulationEngine.

Payload Modes (GimbalMode)
--------------------------
STARE    -- Fixed nadir point-stare; tracks a world XY coordinate.
SCAN     -- Sweeps azimuth sinusoidally at constant elevation (search).
CUED     -- Slaves gimbal to a designated track position (PAY-005).

Gimbal convention
-----------------
Azimuth  : degrees ENU, relative to UAV heading; positive CCW.
           Range: ±PAY_GIMBAL_AZ_RANGE_DEG/2.
Elevation: degrees; 0 = horizon, -90 = straight down (nadir).
           Range: [PAY_GIMBAL_EL_MIN_DEG, PAY_GIMBAL_EL_MAX_DEG].

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
Implements: PAY-001..007, NF-VIZ-006 M4 (FOV cone data exposed).
"""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.base import Entity, EntityType
from sim3dves.payload.detection_engine import DetectionEngine

_D = SimDefaults()


class GimbalMode(Enum):
    """
    Gimbal operating mode (PAY-002).

    STARE : Holds a fixed world XY aim-point (nadir/slant-range).
    SCAN  : Sweeps azimuth at constant depression for area search.
    CUED  : Slaves to the active tracked entity position (PAY-005).
    """
    STARE = auto()
    SCAN = auto()
    CUED = auto()


class OpticalPayload:
    """
    EO/IR sensor payload mounted on a single UAVEntity (PAY-001..007).

    The payload is not an Entity subclass — it is a component owned by
    UAVEntity (composition over inheritance).  SimulationEngine calls
    ``payload.step()`` via the UAV after each UAV kinematic update, then
    calls ``payload.flush_detections()`` to harvest detection events for
    the EventBus (PAY-004).

    Parameters
    ----------
    owner_id : str
        Entity ID of the hosting UAV (for event payload attribution).
    detection_engine : DetectionEngine
        Injected detection strategy (NF-M-002 plugin interface).
    fov_deg : float
        Full cone angle of the sensor (default: PAY_FOV_DEG).
    gimbal_rate_dps : float
        Maximum gimbal slew rate in deg/s (PAY-003).

    Attributes
    ----------
    gimbal_az_deg : float
        Current gimbal azimuth relative to UAV heading (degrees).
    gimbal_el_deg : float
        Current gimbal elevation (degrees; 0 = horizon, -90 = nadir).
    mode : GimbalMode
        Active gimbal FSM mode.
    track_entity_id : str or None
        Entity currently being tracked (CUED mode).
    """

    def __init__(
        self,
        owner_id: str,
        detection_engine: Optional[DetectionEngine] = None,
        fov_deg: float = _D.PAY_FOV_DEG,
        gimbal_rate_dps: float = _D.PAY_GIMBAL_RATE_DPS,
    ) -> None:
        self._owner_id: str = owner_id
        self._detection_engine: DetectionEngine = (
            detection_engine if detection_engine is not None
            else DetectionEngine()
        )
        self._fov_deg: float = float(fov_deg)
        self._gimbal_rate_dps: float = float(gimbal_rate_dps)

        # Gimbal state (PAY-002, PAY-003)
        self.gimbal_az_deg: float = 0.0          # Relative to UAV heading
        self.gimbal_el_deg: float = -90.0        # Start pointing nadir
        self.mode: GimbalMode = GimbalMode.SCAN  # Default: area search

        # STARE aim-point [x, y] in world coordinates
        self._stare_point: Optional[np.ndarray] = None

        # CUED track target
        self.track_entity_id: Optional[str] = None
        self._cued_dwell_remaining_s: float = 0.0

        # Scan sweep state
        self._scan_angle_rad: float = 0.0   # Current sweep phase
        self._scan_speed_dps: float = 20.0  # Azimuth sweep rate (deg/s)

        # Detection event buffer: flushed by engine each step (PAY-004)
        self._pending_detections: List[Dict] = []

        # FOV geometry exposed for visualiser (NF-VIZ-006 M4)
        # Updated by step(); read by DebugPlot._draw_fov_cones()
        self.fov_tip_world: Optional[np.ndarray] = None      # UAV position
        self.fov_axis_world: Optional[np.ndarray] = None     # Unit aim vector (XY)
        self.fov_half_angle_deg: float = self._fov_deg / 2.0

    # ### Public gimbal control API ###

    def command_stare(self, aim_point_xy: np.ndarray) -> None:
        """
        Transition to STARE mode, slewing toward the given world XY point.

        Parameters
        ----------
        aim_point_xy : np.ndarray
            World [x, y] coordinate to stare at.
        """
        self._stare_point = aim_point_xy[:2].astype(float)
        self.mode = GimbalMode.STARE

    def command_scan(self) -> None:
        """Transition to SCAN mode (sinusoidal azimuth sweep)."""
        self.mode = GimbalMode.SCAN

    def command_cued(self, entity_id: str) -> None:
        """
        Transition to CUED mode, slaving the gimbal to the given entity.

        Parameters
        ----------
        entity_id : str
            Entity ID of the track target (PAY-005).
        """
        self.track_entity_id = entity_id
        self._cued_dwell_remaining_s = _D.PAY_CUED_DWELL_S
        self.mode = GimbalMode.CUED

    # ### Main update ###

    def step(
        self,
        uav_position: np.ndarray,
        uav_heading_deg: float,
        entities: List[Entity],
        dt: float,
        structures: Optional[list] = None,
    ) -> None:
        """
        Execute one payload step: articulate gimbal, detect entities in FOV.

        Parameters
        ----------
        uav_position : np.ndarray
            Current UAV 3-D position [x, y, z] (ENU metres).
        uav_heading_deg : float
            UAV heading in degrees (ENU, CCW from East).
        entities : list[Entity]
            All living entities in the world (detection candidates).
        dt : float
            Simulation timestep (s).
        structures : list[AABB], optional
            Opaque structures for LOS occlusion (ENV-003).
        """
        # 1. Articulate gimbal per active mode (PAY-002, PAY-003)
        self._update_gimbal(uav_position, uav_heading_deg, entities, dt)

        # 2. Compute world-space aim axis for FOV queries (PAY-001)
        aim_world = self._compute_aim_vector_world(uav_heading_deg)
        self.fov_tip_world = uav_position.copy()
        self.fov_axis_world = aim_world[:2].copy()  # XY component for 2-D viz

        # 3. Filter entities into FOV (PAY-001)
        in_fov = self._entities_in_fov(uav_position, aim_world, entities)

        # 4. LOS check and P(D) detection (NF-P-004, PAY-004, POL-001)
        detections = self._detection_engine.process_batch(
            observer=uav_position,
            candidates=in_fov,
            structures=structures or [],
        )

        # 5. Buffer detection events for engine harvest (PAY-004)
        for entity, pd, los_clear in detections:
            self._pending_detections.append({
                "uav_id": self._owner_id,
                "target_id": entity.entity_id,
                "target_type": entity.entity_type.name,
                "is_eoi": entity.is_eoi,
                "pd": round(pd, 4),
                "los_clear": los_clear,
                "target_pos": [round(v, 2) for v in entity.position.tolist()],
            })

        # 6. CUED mode: cue UAV orbit after dwell if target is detected EOI
        if self.mode == GimbalMode.CUED and self.track_entity_id is not None:
            self._cued_dwell_remaining_s = max(
                0.0, self._cued_dwell_remaining_s - dt
            )

    def flush_detections(self) -> List[Dict]:
        """
        Return and clear the buffered detection events (PAY-004).

        Called by SimulationEngine.step() after payload.step() completes.

        Returns
        -------
        list[dict]
            Detection event dicts ready for EventBus publication.
        """
        events = self._pending_detections
        self._pending_detections = []
        return events

    # ### Gimbal FSM ###

    def _update_gimbal(
        self,
        uav_pos: np.ndarray,
        uav_heading_deg: float,
        entities: List[Entity],
        dt: float,
    ) -> None:
        """
        Dispatch to mode-specific gimbal articulation handler (PAY-002, PAY-003).
        """
        if self.mode == GimbalMode.STARE:
            self._gimbal_stare(uav_pos)
        elif self.mode == GimbalMode.CUED:
            self._gimbal_cued(uav_pos, entities)
        else:
            self._gimbal_scan(dt)

    def _gimbal_stare(self, uav_pos: np.ndarray) -> None:
        """Slew toward the fixed stare point (STARE mode)."""
        if self._stare_point is None:
            return
        desired_az, desired_el = self._world_point_to_gimbal_angles(
            uav_pos, self._stare_point
        )
        self._slew_to(desired_az, desired_el)

    def _gimbal_cued(
        self, uav_pos: np.ndarray, entities: List[Entity]
    ) -> None:
        """Slave gimbal to the tracked entity position (CUED mode, PAY-005)."""
        if self.track_entity_id is None:
            self.mode = GimbalMode.SCAN
            return
        target = next(
            (e for e in entities if e.entity_id == self.track_entity_id),
            None,
        )
        if target is None or not target.alive:
            # Track lost — revert to scan
            self.track_entity_id = None
            self.mode = GimbalMode.SCAN
            return
        desired_az, desired_el = self._world_point_to_gimbal_angles(
            uav_pos, target.position[:2]
        )
        self._slew_to(desired_az, desired_el)

    def _gimbal_scan(self, dt: float) -> None:
        """Sinusoidal azimuth sweep at constant depression (SCAN mode)."""
        self._scan_angle_rad += math.radians(self._scan_speed_dps) * dt
        az_half = _D.PAY_GIMBAL_AZ_RANGE_DEG / 2.0
        desired_az = az_half * math.sin(self._scan_angle_rad)
        desired_el = (_D.PAY_GIMBAL_EL_MIN_DEG + _D.PAY_GIMBAL_EL_MAX_DEG) / 2.0
        self._slew_to(desired_az, desired_el)

    # ### Gimbal geometry helpers ###

    def _slew_to(self, desired_az: float, desired_el: float) -> None:
        """
        Rate-limited slew toward (desired_az, desired_el) (PAY-003).

        Parameters
        ----------
        desired_az : float
            Desired gimbal azimuth (degrees, relative to UAV heading).
        desired_el : float
            Desired gimbal elevation (degrees).
        """
        # Clamp to gimbal limits (PAY-002, PAY-003)
        az_limit = _D.PAY_GIMBAL_AZ_RANGE_DEG / 2.0
        desired_az = float(np.clip(desired_az, -az_limit, az_limit))
        desired_el = float(
            np.clip(desired_el, _D.PAY_GIMBAL_EL_MIN_DEG, _D.PAY_GIMBAL_EL_MAX_DEG)
        )

        # Rate-limit each axis independently (PAY-003)
        max_step = _D.PAY_GIMBAL_RATE_DPS * 0.1  # dt=0.1 assumed; use param if needed
        def _step(current: float, target: float) -> float:
            delta = target - current
            delta = (delta + 180.0) % 360.0 - 180.0  # wrap to (-180, 180]
            return current + float(np.clip(delta, -max_step, max_step))

        self.gimbal_az_deg = _step(self.gimbal_az_deg, desired_az)
        self.gimbal_el_deg = float(np.clip(
            self.gimbal_el_deg + float(np.clip(
                desired_el - self.gimbal_el_deg, -max_step, max_step
            )),
            _D.PAY_GIMBAL_EL_MIN_DEG,
            _D.PAY_GIMBAL_EL_MAX_DEG,
        ))

    def _world_point_to_gimbal_angles(
        self, uav_pos: np.ndarray, target_xy: np.ndarray
    ) -> Tuple[float, float]:
        """
        Convert world XY target to gimbal (az, el) relative to UAV heading.

        Parameters
        ----------
        uav_pos : np.ndarray
            UAV 3-D position.
        target_xy : np.ndarray
            Target world [x, y] coordinate.

        Returns
        -------
        tuple[float, float]
            (azimuth_deg, elevation_deg) in gimbal frame.
        """
        dx = float(target_xy[0]) - float(uav_pos[0])
        dy = float(target_xy[1]) - float(uav_pos[1])
        dz = -float(uav_pos[2])   # target assumed on ground (z=0)

        world_bearing_rad = math.atan2(dy, dx)   # ENU bearing to target
        # Convert to gimbal azimuth (relative to UAV heading, stored externally)
        # Caller must subtract UAV heading; we return raw world bearing here and
        # the render helper subtracts heading when drawing the cone.
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        elevation_deg = math.degrees(math.atan2(dz, max(horiz_dist, 1e-6)))
        azimuth_world_deg = math.degrees(world_bearing_rad)
        return azimuth_world_deg, elevation_deg

    def _compute_aim_vector_world(
        self, uav_heading_deg: float
    ) -> np.ndarray:
        """
        Compute unit aim vector in world ENU from current gimbal angles.

        Parameters
        ----------
        uav_heading_deg : float
            UAV heading in degrees (ENU, CCW from East).

        Returns
        -------
        np.ndarray
            Unit 3-D ENU aim vector [dx, dy, dz].
        """
        # Absolute azimuth = UAV heading + gimbal azimuth (both in ENU CCW)
        abs_az_rad = math.radians(uav_heading_deg + self.gimbal_az_deg)
        el_rad = math.radians(self.gimbal_el_deg)

        cos_el = math.cos(el_rad)
        dx = cos_el * math.cos(abs_az_rad)
        dy = cos_el * math.sin(abs_az_rad)
        dz = math.sin(el_rad)   # negative for downward-looking elevations
        return np.array([dx, dy, dz])

    # ### FOV membership test ###

    def _entities_in_fov(
        self,
        uav_pos: np.ndarray,
        aim_world: np.ndarray,
        entities: List[Entity],
    ) -> List[Entity]:
        """
        Return entities whose bearing from the UAV is within the FOV cone (PAY-001).

        Uses vectorised NumPy dot products for performance (NF-P-004).

        Parameters
        ----------
        uav_pos : np.ndarray
            UAV 3-D position.
        aim_world : np.ndarray
            Unit 3-D aim vector.
        entities : list[Entity]
            Candidate entities.

        Returns
        -------
        list[Entity]
            Entities inside the FOV cone, within PAY_FOOTPRINT_MAX_M.
        """
        if not entities:
            return []

        half_angle_rad = math.radians(self._fov_deg / 2.0)
        cos_half = math.cos(half_angle_rad)

        # Build position matrix (n, 3) and compute direction vectors
        positions = np.array([e.position for e in entities])   # (n, 3)
        dirs = positions - uav_pos                             # (n, 3)
        dists = np.linalg.norm(dirs, axis=1)                   # (n,)

        # Exclude UAV itself (dist ≈ 0) and entities beyond max footprint
        valid = (dists > _D.PAY_DETECT_MIN_RANGE_M) & (
            dists < _D.PAY_FOOTPRINT_MAX_M
        )

        # Normalise direction vectors (avoid divide-by-zero)
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_dirs = dirs / safe_dists[:, np.newaxis]           # (n, 3)

        # Dot product with aim vector: cos(angle between aim and entity)
        dots = unit_dirs @ aim_world                           # (n,)
        in_cone = dots >= cos_half

        return [entities[i] for i in range(len(entities)) if valid[i] and in_cone[i]]
