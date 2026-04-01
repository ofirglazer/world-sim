"""
sim3dves.entities.pedestrian
=============================
Pedestrian entity with terrain-locked kinematics and speed envelope and social-force avoidance.

M2 additions
------------
PED-003: Simplified social force model applied in _update_behavior() when
a StepContext with neighbors is present.  The force vector is computed
as a linear repulsion from each nearby pedestrian, then velocity is
re-normalized so the speed envelope is preserved.

Design Pattern: Template Method (inherits Entity.step() skeleton).
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-005: Inline comments throughout.
Implements: PED-001 (speed), PED-002 (nav stub), PED-003 (social force),
            PED-004 (EOI), Req-7 (terrain lock).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.entities.base import Entity, EntityState, EntityType
from sim3dves.entities.context import StepContext

_D = SimDefaults()


class PedestrianEntity(Entity):
    """
    Surface-bound pedestrian with heading-noise locomotion and social force.

    Invariants enforced every step
    --------------------------------
    1. ``velocity[2]`` is always 0.0 (terrain lock — Req-7).
    2. ``position[2]`` is always the terrain elevation (flat-terrain elevation — Req-7).
    3. XY speed stays within [PED_SPEED_MIN_MPS, PED_SPEED_MAX_MPS].

    Design Pattern: Template Method (inherits Entity.step() skeleton).

Parameters
    ----------
    entity_id : str
        Unique identifier.
    position : np.ndarray
        Initial 3-D position.  Z is overridden to terrain elevation (Req-7).
    velocity : np.ndarray
        Initial velocity; Z component zeroed; XY normalised to speed_mps.
    heading : float
        Initial heading (degrees).
    is_eoi : bool
        Entity of Interest flag (PED-004).
    signature : float
        Optical/thermal contrast in [0, 1] (VEH-006 schema).
    speed_mps : float, optional
        Fixed target speed.  If None, sampled from [MIN, MAX] (PED-001).

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
        # Terrain lock at construction: Z = 0 (flat terrain, Req-7)
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

        # Target speed within the pedestrian envelope (PED-001)
        self._target_speed_mps: float = (
            float(speed_mps) if speed_mps is not None
            else float(np.random.uniform(
                _D.PED_SPEED_MIN_MPS, _D.PED_SPEED_MAX_MPS
            ))
        )

        # Ensure initial velocity already satisfies constraints
        self._enforce_terrain_lock()
        self._normalise_speed()

    # ### Template Method hooks ###

    def _update_behavior(
            self, dt: float, context: Optional[StepContext] = None
    ) -> None:
        """
        Apply heading perturbation and social-force repulsion (PED-001..003).

        Heading perturbation: small Gaussian angular rotation preserving
        speed magnitude (unlike additive Cartesian noise which grows speed).

        Social force (PED-003): linear repulsion from nearby pedestrians.
        Applied as a velocity delta; _normalise_speed() in _update_kinematics
        restores the target speed envelope.

        Parameters
        ----------
        dt : float
            Simulation timestep.
        context : StepContext, optional
            Neighbor snapshot. Social force applied only when not None.
        """
        # 1. Angular heading perturbation (PED-001 speed-preserving noise)
        delta_rad = float(np.random.normal(0.0, _D.PED_VELOCITY_NOISE_STD))
        self._rotate_velocity_xy(delta_rad)

        # 2. Social force repulsion from nearby pedestrians (PED-003)
        if context is not None and context.neighbors:
            self._apply_social_force(context.neighbors)

        self.state = EntityState.MOVING

    def _update_kinematics(self, dt: float) -> None:
        """
        Euler integrate XY position; enforce terrain lock and speed envelope.

        Terrain lock (Req-7): Z velocity is zeroed; Z position held at 0.
        Speed normalisation (PED-001): re-clamps after social force delta.
        PED-001: speed is re-clamped after each step to guard against
        floating-point drift accumulation over long runs.
        """
        # Re-enforce invariants (defensive — behavior hook may have perturbed)
        self._enforce_terrain_lock()  # Z velocity = 0 (Req-7)
        self._normalise_speed()  # Speed envelope after social force

        # Euler integration — XY only
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        # Z remains at terrain elevation (flat = 0.0, Req-7)

    # ### Private helpers ###

    def _apply_social_force(self, neighbours: list) -> None:
        """
        Apply linear repulsion from pedestrian neighbors (PED-003).

        Force magnitude decreases linearly from SOCIAL_REPULSION_K at contact
        to 0 at SOCIAL_RADIUS_M.  Normalisation happens in _update_kinematics.

        Parameters
        ----------
        neighbours : list of Entity
            Nearby entities from StepContext.
        """
        fx, fy = 0.0, 0.0
        for neighbour in neighbours:
            # Social force applies only between pedestrians
            if neighbour.entity_type is not EntityType.PEDESTRIAN:
                continue
            dx = float(self.position[0] - neighbour.position[0])
            dy = float(self.position[1] - neighbour.position[1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.1 or dist >= _D.SOCIAL_RADIUS_M:
                continue  # Min distance guard prevents division by zero
            # Linear repulsion: force magnitude = K * (1 - d/R)
            magnitude = _D.SOCIAL_REPULSION_K * (1.0 - dist / _D.SOCIAL_RADIUS_M)
            fx += (dx / dist) * magnitude
            fy += (dy / dist) * magnitude
        self.velocity[0] += fx
        self.velocity[1] += fy

    def _enforce_terrain_lock(self) -> None:
        """
        Zero out the Z velocity component (Req-7: terrain lock).

        Pedestrians are surface-bound — any Z velocity introduced by
        construction arguments or future behavior hooks is discarded here.
        """
        self.velocity[2] = 0.0

    def _normalise_speed(self) -> None:
        """
        Re-scale XY velocity to exactly ``_target_speed_mps``  (PED-001).

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
        Rotate XY velocity by *delta_rad* radians using a 2-D rotation matrix.

        Preserves vector magnitude exactly — unlike additive Cartesian noise.
        """
        cos_d = math.cos(delta_rad)
        sin_d = math.sin(delta_rad)
        vx, vy = float(self.velocity[0]), float(self.velocity[1])
        self.velocity[0] = cos_d * vx - sin_d * vy
        self.velocity[1] = sin_d * vx + cos_d * vy
