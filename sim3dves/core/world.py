"""
sim3dves.core.world
===================
Immutable spatial model of the simulation environment.

M2 addition
-----------
``road_network: Optional[RoadNetwork]`` field added to World.
World now acts as a Facade over Terrain, AABB structures, NFZ volumes,
*and* the road network — providing a single authoritative spatial API.


Implements: ENV-001 through ENV-007.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-005: Inline comments throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sim3dves.maps.road_network import RoadNetwork

# ### Structure primitives ###


@dataclass(frozen=True)
class AABB:
    """
    Axis-Aligned Bounding Box obstacle (ENV-003).

    Represents an opaque obstacle that blocks line-of-sight
    and prevents entity traversal.

    Attributes
    ----------
    x, y : float
        South-West corner position (ENU metres).
    width, depth : float
        Extents along X and Y axes (m).
    height : float
        Structure height above terrain (m).
    """
    x: float  # South-West corner X (m, ENU East axis)
    y: float  # South-West corner Y (m, ENU North axis)
    width: float  # Extent along X (m)
    depth: float  # Extent along Y (m)
    height: float = 0.0  # Structure height above ground (m)

    def contains_xy(self, position: np.ndarray) -> bool:
        """Return True if (x, y) of *position* falls inside this footprint."""
        return (
            self.x <= position[0] <= self.x + self.width
            and self.y <= position[1] <= self.y + self.depth
        )


@dataclass(frozen=True)
class NFZCylinder:
    """
    Cylindrical No-Fly Zone (ENV-005).

    Any UAV whose 3-D position falls inside this volume
    is in violation of FLR-001.

    Attributes
    ----------
    center_xy : np.ndarray
        [x, y] center of the cylinder (m).
    radius_m : float
        Horizontal radius (m).
    alt_max_m : float
        AGL altitude ceiling of this NFZ (m).
    """
    center_xy: np.ndarray  # (x, y) center of the cylinder (m)
    radius_m: float  # Horizontal radius (m)
    alt_max_m: float  # AGL altitude ceiling of the NFZ (m)

    def contains(self, position: np.ndarray) -> bool:
        """Return True if *position* violates this NFZ volume."""
        dist_xy = float(np.linalg.norm(position[:2] - self.center_xy))
        return dist_xy <= self.radius_m and position[2] <= self.alt_max_m


# ### Terrain model ###

class Terrain:
    """
    Ground elevation model (ENV-001, ENV-002).

    Default implementation: flat plane at Z = 0.
    Override ``elevation_at()`` to load a DEM raster (future extension).

    GPU note (NF-HA-001): batch elevation lookups can be vectorized with
    CuPy when a raster DEM is loaded — the interface is intentionally
    array-friendly for that reason.
    """

    def __init__(self, extent: np.ndarray) -> None:
        # extent is (X_max, Y_max) in metres — stored for bounds validation
        self._extent: np.ndarray = extent

    def elevation_at(self, xy: np.ndarray) -> float:
        """
        Return terrain elevation at position *xy* = [x, y] in metres above datum.

        Parameters
        ----------
        xy : np.ndarray
            2-D horizontal position [x, y].

        Returns
        -------
        float
            0.0 for flat terrain (DEM placeholder for M3+).
        """
        # Flat-terrain default — replace with DEM bilinear interpolation in M2+
        return 0.0

    @property
    def extent(self) -> np.ndarray:
        """World extent [X_max, Y_max] in metres."""
        return self._extent


# ### World facade ###

class World:
    """
    Spatial Facade over terrain, structures, NFZ volumes, and road network.

    M2: added ``road_network`` field (ENV-006, VEH-003).

    Parameters
    ----------
    extent : np.ndarray
        [X_max, Y_max] world footprint in metres (ENV-001).
    alt_floor_m : float
        UAV minimum AGL (FLR-002).
    alt_ceil_m : float
        UAV maximum AGL (FLR-003).
    structures : list[AABB], optional
        Opaque obstacles for LOS / occlusion checks (ENV-003).
    nfz_cylinders : list[NFZCylinder], optional
        No-fly zone volumes (ENV-005).
    road_network : RoadNetwork, optional
        Road graph for vehicle navigation (ENV-006, VEH-003).
    """

    def __init__(
        self,
        extent: np.ndarray,
        alt_floor_m: float = 0.0,
        alt_ceil_m: float = 500.0,
        structures: Optional[List[AABB]] = None,
        nfz_cylinders: Optional[List[NFZCylinder]] = None,
        road_network: Optional[RoadNetwork] = None,
    ) -> None:
        self.terrain: Terrain = Terrain(extent)
        self.extent: np.ndarray = extent.astype(float)
        self.alt_floor_m: float = float(alt_floor_m)
        self.alt_ceil_m: float = float(alt_ceil_m)
        self.structures: List[AABB] = structures or []
        self.nfz_cylinders: List[NFZCylinder] = nfz_cylinders or []
        self.road_network: Optional[RoadNetwork] = road_network  # ENV-006

    # ### Boundary queries ###

    def in_bounds(self, position: np.ndarray) -> bool:
        """Return True if *position* XY is within the world footprint."""
        return (
            0.0 <= position[0] <= self.extent[0]
            and 0.0 <= position[1] <= self.extent[1]
        )

    def in_nfz(self, position: np.ndarray) -> bool:
        """Return True if *position* violates any configured NFZ (FLR-001)."""
        return any(nfz.contains(position) for nfz in self.nfz_cylinders)

    def occluded_by_structure(self, position: np.ndarray) -> bool:
        """Return True if *position* (x, y) is inside a structure footprint."""
        return any(s.contains_xy(position) for s in self.structures)

    # ### Terrain helpers ###

    def terrain_elevation(self, xy: np.ndarray) -> float:
        """Return terrain elevation at (x, y) — delegates to Terrain model."""
        return self.terrain.elevation_at(xy)

    def snap_to_terrain(self, position: np.ndarray) -> np.ndarray:
        """
        Return copy of position with Z set to terrain elevation (Req-7).

        Used by entity spawners to ensure ground entities start on the
        terrain surface rather than at an arbitrary Z value.

        Parameters
        ----------
        position : np.ndarray
            3-D position [x, y, z_ignored].

        Returns
        -------
        np.ndarray
            New position with z = terrain_elevation(x, y).
        """
        snapped = position.astype(float).copy()
        snapped[2] = self.terrain.elevation_at(position[:2])
        return snapped
