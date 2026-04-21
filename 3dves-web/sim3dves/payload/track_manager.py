"""
sim3dves.payload.track_manager
================================
TrackManager: per-entity track lifecycle with Kalman-filter state estimation,
quality grading, and autonomous re-acquisition (M5).

Design Pattern: Registry — TrackManager owns a dict of TrackState keyed by
entity ID, updated each step by the SimulationEngine.

Kalman Filter Model
-------------------
Constant-velocity 4-state model: x = [x, y, vx, vy].

    Transition matrix F (dt = simulation timestep):
        [1  0  dt  0 ]
        [0  1  0   dt]
        [0  0  1   0 ]
        [0  0  0   1 ]

    Measurement matrix H (position only):
        [1  0  0  0]
        [0  1  0  0]

    Process noise Q:  σ_q² * diag([dt⁴/4, dt⁴/4, dt², dt²])
        Represents unmodelled acceleration (discrete Wiener model).

    Measurement noise R: σ_r² * I₂

Track Quality (M5)
------------------
Quality is evaluated each step based on consecutive hit/miss counts:

    HIGH   — ≥ TRK_HIGH_HIT_COUNT  consecutive detections
    MEDIUM — ≥ TRK_MEDIUM_HIT_COUNT consecutive detections (but < HIGH)
    LOW    — track active but recently missed
    LOST   — ≥ TRK_MAX_MISS_STEPS consecutive misses → track removed

Events published (via SimulationEngine EventBus):
    TRACK_ACQUIRED     — fired once when a track first reaches MEDIUM quality
                         (LOW→MEDIUM transition only).
    TRACK_QUALITY_HIGH — fired once when a track first reaches HIGH quality
                         (MEDIUM→HIGH transition only).  BUG-011 fix.
    TRACK_LOST         — fired when a track is removed after TRK_MAX_MISS_STEPS.

BUG-011 fix
-----------
Added ``on_track_quality_high`` optional callback and ``newly_high`` promotion
list in ``step()``.  This separates the MEDIUM-acquisition signal (used by
loggers and general observers) from the HIGH-confidence signal (used by the
HighQualityEoiCueingPolicy).

The two contracts are now independently satisfied:
  - test_m5.py : ``on_track_acquired`` fires exactly once at LOW→MEDIUM;
                 it does NOT fire again at MEDIUM→HIGH.
  - test_runner.py : the cueing policy reacts to ``on_track_quality_high``
                     which carries quality == HIGH.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Registry pattern.
Implements: M5 (TrackManager, Kalman filter, track quality, re-acquisition).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.base import Entity

_D = SimDefaults()


# ---------------------------------------------------------------------------
# Track quality enumeration
# ---------------------------------------------------------------------------

class TrackQuality(Enum):
    """
    Track confidence tier (M5).

    Determined each step by comparing consecutive hit / miss counts against
    the TRK_* threshold constants in SimDefaults.
    """
    LOW    = auto()   # Active but recently missed
    MEDIUM = auto()   # TRK_MEDIUM_HIT_COUNT consecutive detections
    HIGH   = auto()   # TRK_HIGH_HIT_COUNT   consecutive detections


# ---------------------------------------------------------------------------
# Per-track state dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """
    Kalman-filter state for one tracked entity (M5).

    Attributes
    ----------
    entity_id : str
        The entity being tracked.
    x : np.ndarray
        State vector [x_m, y_m, vx_mps, vy_mps], shape (4,).
    P : np.ndarray
        State covariance matrix, shape (4, 4).
    quality : TrackQuality
        Current confidence tier.
    age_steps : int
        Total steps this track has been active.
    hit_count : int
        Consecutive detection hits (resets to 0 on miss).
    miss_count : int
        Consecutive detection misses (resets to 0 on hit).
    is_eoi : bool
        True if the tracked entity was flagged as EOI at last detection.
    _high_fired : bool
        Internal guard: True once the MEDIUM→HIGH callback has fired so it
        is never re-fired if quality drops and recovers.  (BUG-011 fix)
    """
    entity_id: str
    x: np.ndarray                       # (4,) state vector
    P: np.ndarray                       # (4, 4) covariance
    quality: TrackQuality = TrackQuality.LOW
    age_steps: int = 0
    hit_count: int = 0
    miss_count: int = 0
    is_eoi: bool = False
    _high_fired: bool = False           # BUG-011: guard for MEDIUM→HIGH callback

    @property
    def position_xy(self) -> np.ndarray:
        """Current estimated position [x, y] in world metres."""
        return self.x[:2].copy()

    @property
    def velocity_xy(self) -> np.ndarray:
        """Current estimated velocity [vx, vy] in m/s."""
        return self.x[2:].copy()

    @property
    def position_covariance(self) -> np.ndarray:
        """2×2 position sub-block of the full covariance matrix."""
        return self.P[:2, :2].copy()


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------

# Callback type: (entity_id, TrackState) → None
TrackCallback = Callable[[str, "TrackState"], None]


class TrackManager:
    """
    Manages the lifecycle of all entity tracks (M5).

    Called once per step by SimulationEngine.  For each active track:
    1. **Predict**: propagates the Kalman state forward by dt.
    2. **Update / miss**: if the entity was detected this step, runs the
       Kalman measurement update; otherwise increments miss_count.
    3. **Quality grading**: promotes/demotes quality based on
       consecutive hit / miss counts.
    4. **Track removal**: LOST tracks (miss_count ≥ TRK_MAX_MISS_STEPS)
       trigger a TRACK_LOST callback and are removed from the registry.
    5. **New tracks**: entities detected but not yet tracked are
       initialised.

    Parameters
    ----------
    on_track_acquired : TrackCallback, optional
        Called when a track first reaches MEDIUM quality (LOW→MEDIUM).
        Fires exactly once per transition; does NOT fire again at HIGH.
    on_track_quality_high : TrackCallback, optional
        Called when a track first reaches HIGH quality (MEDIUM→HIGH).
        BUG-011 fix: separate from ``on_track_acquired`` so the cueing
        policy can react only to HIGH-confidence tracks while TRACK_ACQUIRED
        retains its original MEDIUM-quality semantics.
    on_track_lost : TrackCallback, optional
        Called when a track is removed (TRACK_LOST).

    Attributes
    ----------
    active_tracks : dict[str, TrackState]
        All currently active tracks, keyed by entity ID.
        Read by the visualiser to draw ellipses (NF-VIZ-006 M5).
    """

    # Kalman model matrices (recomputed per step because dt may vary)
    _H: np.ndarray = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=float)  # (2, 4)

    def __init__(
        self,
        on_track_acquired:     Optional[TrackCallback] = None,
        on_track_quality_high: Optional[TrackCallback] = None,  # BUG-011 fix
        on_track_lost:         Optional[TrackCallback] = None,
    ) -> None:
        self.active_tracks: Dict[str, TrackState] = {}
        self._on_acquired:      Optional[TrackCallback] = on_track_acquired
        self._on_quality_high:  Optional[TrackCallback] = on_track_quality_high
        self._on_lost:          Optional[TrackCallback] = on_track_lost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        detected_ids: Set[str],
        entities: List[Entity],
        dt: float,
    ) -> None:
        """
        Advance all tracks by one simulation step (M5).

        Parameters
        ----------
        detected_ids : set[str]
            Entity IDs confirmed detected this step
            (from ``SimulationEngine.step_detections``).
        entities : list[Entity]
            All living entities; used to initialise new tracks and to
            retrieve measured positions.
        dt : float
            Simulation timestep (s).
        """
        entity_map: Dict[str, Entity] = {e.entity_id: e for e in entities}

        # 1. Predict all existing tracks
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        for track in self.active_tracks.values():
            track.x = F @ track.x
            track.P = F @ track.P @ F.T + Q
            track.age_steps += 1

        # 2. Update or miss each existing track
        lost_ids: List[str] = []
        newly_medium: List[str] = []
        newly_high: List[str] = []   # BUG-011 fix: tracks reaching HIGH quality
        for eid, track in self.active_tracks.items():
            if eid in detected_ids and eid in entity_map:
                z = entity_map[eid].position[:2].astype(float)
                self._kf_update(track, z)
                track.hit_count  += 1
                track.miss_count  = 0
                track.is_eoi      = entity_map[eid].is_eoi
                prev_quality      = track.quality
                track.quality     = self._grade(track)

                # LOW→MEDIUM transition: fire TRACK_ACQUIRED callback once
                if (prev_quality == TrackQuality.LOW
                        and track.quality == TrackQuality.MEDIUM):
                    newly_medium.append(eid)

                # MEDIUM→HIGH transition: fire TRACK_QUALITY_HIGH callback once.
                # _high_fired guard ensures this fires at most once per track
                # lifetime even if quality drops to LOW and recovers.
                # BUG-011 fix.
                if (track.quality == TrackQuality.HIGH
                        and not track._high_fired):
                    track._high_fired = True
                    newly_high.append(eid)
            else:
                track.miss_count += 1
                track.hit_count   = 0
                track.quality     = TrackQuality.LOW
                if track.miss_count >= _D.TRK_MAX_MISS_STEPS:
                    lost_ids.append(eid)

        # 3. Fire TRACK_ACQUIRED callbacks (LOW→MEDIUM)
        for eid in newly_medium:
            if self._on_acquired is not None:
                self._on_acquired(eid, self.active_tracks[eid])

        # 4. Fire TRACK_QUALITY_HIGH callbacks (MEDIUM→HIGH) — BUG-011 fix
        for eid in newly_high:
            if self._on_quality_high is not None:
                self._on_quality_high(eid, self.active_tracks[eid])

        # 5. Remove lost tracks and fire TRACK_LOST callbacks
        for eid in lost_ids:
            track = self.active_tracks.pop(eid)
            if self._on_lost is not None:
                self._on_lost(eid, track)

        # 6. Initialise tracks for newly detected entities
        R = self._build_R()
        for eid in detected_ids:
            if eid in self.active_tracks:
                continue
            if eid not in entity_map:
                continue
            entity = entity_map[eid]
            # Initialise state from measurement; velocity = 0
            x0 = np.array([
                entity.position[0], entity.position[1],
                entity.velocity[0], entity.velocity[1],
            ], dtype=float)
            # Initial covariance: high position uncertainty (from R), low vel
            P0 = np.diag([
                _D.TRK_MEAS_NOISE_STD ** 2,
                _D.TRK_MEAS_NOISE_STD ** 2,
                (_D.TRK_MEAS_NOISE_STD * 2.0) ** 2,
                (_D.TRK_MEAS_NOISE_STD * 2.0) ** 2,
            ])
            self.active_tracks[eid] = TrackState(
                entity_id=eid,
                x=x0,
                P=P0,
                hit_count=1,
                is_eoi=entity.is_eoi,
            )

    # ------------------------------------------------------------------
    # Kalman filter helpers
    # ------------------------------------------------------------------

    def _build_F(self, dt: float) -> np.ndarray:
        """
        Build the constant-velocity state transition matrix for timestep dt.

        Parameters
        ----------
        dt : float
            Simulation timestep (s).

        Returns
        -------
        np.ndarray
            Shape (4, 4) transition matrix.
        """
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Build the process noise covariance matrix (discrete Wiener model).

        Uses the factored form Q = σ_q² * (Gx*Gxᵀ + Gy*Gyᵀ) where
        Gx = [dt²/2, 0, dt, 0]ᵀ and Gy = [0, dt²/2, 0, dt]ᵀ.
        This guarantees Q is positive semi-definite for all dt.

        Parameters
        ----------
        dt : float
            Simulation timestep (s).

        Returns
        -------
        np.ndarray
            Shape (4, 4) process noise matrix (guaranteed PSD).
        """
        sig2 = _D.TRK_PROCESS_NOISE_STD ** 2
        dt2  = dt * dt
        Gx = np.array([dt2 / 2.0, 0.0,       dt, 0.0], dtype=float)
        Gy = np.array([0.0,       dt2 / 2.0, 0.0, dt], dtype=float)
        return sig2 * (np.outer(Gx, Gx) + np.outer(Gy, Gy))

    def _build_R(self) -> np.ndarray:
        """
        Build the measurement noise covariance matrix (2×2 diagonal).

        Returns
        -------
        np.ndarray
            Shape (2, 2) measurement noise matrix.
        """
        sig2 = _D.TRK_MEAS_NOISE_STD ** 2
        return np.diag([sig2, sig2])

    def _kf_update(self, track: TrackState, z: np.ndarray) -> None:
        """
        Apply the Kalman measurement update in-place (M5).

        Uses the standard KF equations:
            y  = z − H x̂        (innovation)
            S  = H P Hᵀ + R      (innovation covariance)
            K  = P Hᵀ S⁻¹       (Kalman gain)
            x̂  = x̂ + K y        (state update)
            P  = (I − K H) P     (covariance update, Joseph form for stability)

        Parameters
        ----------
        track : TrackState
            Track to update (modified in place).
        z : np.ndarray
            Measurement vector [x, y] in world metres, shape (2,).
        """
        R = self._build_R()
        H = self._H
        y = z - H @ track.x                          # innovation (2,)
        S = H @ track.P @ H.T + R                    # innovation covariance (2,2)
        K = track.P @ H.T @ np.linalg.inv(S)        # Kalman gain (4,2)
        track.x = track.x + K @ y                   # state update
        I_KH    = np.eye(4) - K @ H
        # Joseph form: numerically stable even when K is not optimal
        track.P = I_KH @ track.P @ I_KH.T + K @ R @ K.T  # (4,4)

    # ------------------------------------------------------------------
    # Quality grading
    # ------------------------------------------------------------------

    def _grade(self, track: TrackState) -> TrackQuality:
        """
        Assign a quality tier from current consecutive hit count.

        Parameters
        ----------
        track : TrackState
            Track to evaluate.

        Returns
        -------
        TrackQuality
            Computed quality tier.
        """
        if track.hit_count >= _D.TRK_HIGH_HIT_COUNT:
            return TrackQuality.HIGH
        if track.hit_count >= _D.TRK_MEDIUM_HIT_COUNT:
            return TrackQuality.MEDIUM
        return TrackQuality.LOW

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_track(self, entity_id: str) -> Optional[TrackState]:
        """
        Return the TrackState for the given entity, or None if not tracked.

        Parameters
        ----------
        entity_id : str
            Entity to look up.

        Returns
        -------
        TrackState or None
        """
        return self.active_tracks.get(entity_id)

    def eoi_tracks(self) -> List[TrackState]:
        """Return all active tracks for EOI-flagged entities."""
        return [t for t in self.active_tracks.values() if t.is_eoi]
