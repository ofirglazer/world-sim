"""
sim3dves.entities.pedestrian
=============================
Pedestrian entity with terrain-locked kinematics and speed envelope.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-005: Inline comments throughout.
Implements: PED-001 (speed envelope), PED-002 (waypoint nav stub),
            PED-003 (social force stub), PED-004 (EOI flag).
Req-7: Z coordinate adheres to terrain — never randomly generated.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.base import Entity, EntityState, EntityType

_DEFAULTS = SimDefaults()


class PedestrianEntity(Entity):
    """
    Ground pedestrian with terrain-surface kinematics.

    Key invariants maintained at every step:
    1. ``velocity[2]`` is always 0.0 — pedestrians never move vertically.
    2. ``position[2]`` is always the terrain elevation (0.0 for flat terrain).
    3. XY speed stays within [PED_SPEED_MIN_MPS, PED_SPEED_MAX_MPS].

    Design Pattern: Template Method (inherits Entity.step() skeleton).

    FIX vs original:
    ----------------
    1. ``velocity += np.random.randn(3) * 0.1`` grew velocity unboundedly —
       replaced with angle-perturbation that preserves magnitude (PED-001).
    2. ``np.random.randn(3)`` added Z velocity — terrain lock violated (Req-7).
    3. No speed clamping — pedestrians could reach arbitrary speeds.
    4. No ``EntityType`` enum — was passed as positional arg to ``super()``.
    5. No EOI flag, no signature field.
    6. No FSM state transitions.
    """

    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        heading: float = 0.0,
        is_eoi: bool = False,
        signature: float = 0.8,
        speed_mps: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        entity_id : str
            Unique identifier.
        position : np.ndarray
            Initial 3-D position [x, y, z].  Z is overridden to terrain
            elevation immediately — callers should pass 0.0 or use
            ``World.snap_to_terrain()`` before construction (Req-7).
        velocity : np.ndarray
            Initial 3-D velocity. Z component is zeroed on construction.
            XY magnitude is normalized to *speed_mps*.
        heading : float
            Initial heading in degrees.
        is_eoi : bool
            Marks this entity as an Entity of Interest (PED-004).
        signature : float
            Optical/thermal contrast factor in [0, 1].
        speed_mps : float, optional
            Fixed target speed (m/s).  If None, sampled uniformly from
            [PED_SPEED_MIN_MPS, PED_SPEED_MAX_MPS] (PED-001).
        """
        # Terrain lock: Z is always elevation at construction, never random
        terrain_pos = position.astype(float).copy()
        terrain_pos[2] = 0.0  # Flat terrain default (Req-7)

        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.PEDESTRIAN,
            position=terrain_pos,
            velocity=velocity.astype(float),
            heading=heading,
            is_eoi=is_eoi,
            signature=signature,
        )

        # Assign target speed within the pedestrian envelope (PED-001)
        self._target_speed_mps: float = (
            float(speed_mps)
            if speed_mps is not None
            else float(np.random.uniform(
                _DEFAULTS.PED_SPEED_MIN_MPS,
                _DEFAULTS.PED_SPEED_MAX_MPS,
            ))
        )

        # Ensure initial velocity already satisfies constraints
        self._enforce_terrain_lock()
        self._normalise_speed()

    # ### Template Method hooks ###

    def _update_behavior(self, dt: float) -> None:
        """
        Perturb heading direction and update FSM state.

        PED-002 (waypoint navigation): stub — full A* nav in M2+.
        PED-003 (social force):        stub — repulsion in M2+.

        The perturbation rotates the XY velocity by a small random angle
        sampled from N(0, PED_VELOCITY_NOISE_STD).  This preserves speed
        magnitude — unlike the original ``velocity += randn(3) * 0.1``
        which caused unbounded drift.
        """
        # Sample a small angular perturbation (radians)
        delta_rad = float(np.random.normal(
            0.0, _DEFAULTS.PED_VELOCITY_NOISE_STD
        ))
        self._rotate_velocity_xy(delta_rad)
        self.state = EntityState.MOVING

    def _update_kinematics(self, dt: float) -> None:
        """
        Integrate XY velocity; Z is held at terrain elevation.

        PED-001: speed is re-clamped after each step to guard against
        floating-point drift accumulation over long runs.
        """
        # Re-enforce invariants (defensive — behavior hook may have perturbed)
        self._enforce_terrain_lock()
        self._normalise_speed()

        # Euler integration — XY only
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        # Z remains at terrain elevation (flat = 0.0)

    # ### Private helpers ###

    def _enforce_terrain_lock(self) -> None:
        """
        Zero out the Z velocity component.

        Pedestrians are surface-bound — any Z velocity introduced by
        construction arguments or future behavior hooks is discarded here.
        """
        self.velocity[2] = 0.0

    def _normalise_speed(self) -> None:
        """
        Re-scale XY velocity to exactly ``_target_speed_mps``.

        If XY magnitude is negligible (< 1e-9), assign a random direction
        so the entity does not freeze in place.
        """
        xy_mag = float(np.linalg.norm(self.velocity[:2]))
        if xy_mag < 1e-9:
            # Dead-stop recovery: pick a random heading
            angle = float(np.random.uniform(0.0, 2.0 * math.pi))
            self.velocity[0] = math.cos(angle) * self._target_speed_mps
            self.velocity[1] = math.sin(angle) * self._target_speed_mps
        else:
            # Scale to target speed without changing direction
            scale = self._target_speed_mps / xy_mag
            self.velocity[0] *= scale
            self.velocity[1] *= scale

    def _rotate_velocity_xy(self, delta_rad: float) -> None:
        """
        Rotate XY velocity vector by *delta_rad* radians (2-D rotation matrix).

        This preserves the vector magnitude precisely, unlike adding noise
        to the Cartesian components directly.
        """
        cos_d = math.cos(delta_rad)
        sin_d = math.sin(delta_rad)
        vx, vy = float(self.velocity[0]), float(self.velocity[1])
        self.velocity[0] = cos_d * vx - sin_d * vy
        self.velocity[1] = sin_d * vx + cos_d * vy
