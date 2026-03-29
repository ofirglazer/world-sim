"""
sim3dves.viz.debug_plot
=======================
Real-time 2-D top-down matplotlib debug visualiser.

M2 additions (NF-VIZ-006)
--------------------------
* Road network overlay: edges as grey lines, nodes as small dots.
* Per-type marker shapes: pedestrian = circle, wheeled = square,
  tracked = diamond, UAV = triangle.
* ``road_network`` accepted in constructor for automatic road drawing.

Incremental feature roadmap (NF-VIZ-006)
-----------------------------------------
M1 : pedestrian positions, heading arrows, EOI markers, legend, clock.
M2 : vehicle positions, road network overlay (this file).
M3 : NFZ circles, geofence boundary.
M4 : Payload FOV cone projected onto ground plane.
M5 : Active track centroids and covariance ellipses.
M6 : Road network graph overlay (full detail).

NF-CE-001..005 compliant. Implements: NF-VIZ-001..006.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sim3dves.entities.base import Entity, EntityType
from sim3dves.maps.road_network import RoadNetwork

# ### Colour palette ###
_TYPE_COLOURS: Dict[EntityType, str] = {
    EntityType.PEDESTRIAN:      "#2196F3",   # Blue
    EntityType.WHEELED_VEHICLE: "#4CAF50",   # Green
    EntityType.TRACKED_VEHICLE: "#FF9800",   # Orange
    EntityType.UAV:             "#9C27B0",   # Purple
}

# ── Marker shapes per type (NF-VIZ-002, M2) ──────────────────────────────────
_TYPE_MARKERS: Dict[EntityType, str] = {
    EntityType.PEDESTRIAN:      "o",  # Circle
    EntityType.WHEELED_VEHICLE: "s",  # Square
    EntityType.TRACKED_VEHICLE: "D",  # Diamond
    EntityType.UAV:             "^",  # Triangle
}

_EOI_EDGE_COLOUR: str = "#F44336"   # Red ring around EOI entities
_DEAD_COLOUR: str = "#BDBDBD"       # Grey for dead entities
_ROAD_COLOUR: str = "#9E9E9E"       # Road edge colour
_NODE_COLOUR: str = "#757575"       # Road node dot colour
_HEADING_ALPHA: float = 0.70        # Arrow transparency


class DebugPlot:
    """
    Interactive top-down 2-D visualiser backed by matplotlib.

    Designed to be called once per simulation step inside the run loop.
    Rendering is intentionally lightweight — the axes are fully cleared
    (``cla()``) and redrawn each frame.  For smoother animation in later
    milestones, consider blitting or ``FuncAnimation``.

    Parameters
    ----------
    world_x : float
        World extent along East (X) axis in metres.
    world_y : float
        World extent along North (Y) axis in metres.
    road_network : RoadNetwork, optional
        If provided, road edges and nodes are drawn each frame (M2).
    title : str
        Window title string.


    Incremental feature roadmap
    ---------------------------
    M1 - Pedestrian positions, headings, EOI markers, legend, clock.
    M2 - UAV positions and altitude annotation.
    M3 - NFZ circles, geofence boundary.
    M4 - Payload FOV cone projected onto ground plane.
    M5 - Active track centroids and covariance ellipses.
    M6 - Road network graph overlay.
    """

    def __init__(
        self,
        world_x: float,
        world_y: float,
        road_network: Optional[RoadNetwork] = None,
        title: str = "3DVES — Debug View",
    ) -> None:

        # Guard against re-enabling interactive mode on repeated construction
        if not plt.isinteractive():
            plt.ion()

        self._fig, self._ax = plt.subplots(figsize=(9, 9))
        # Set window title safely (backend may not support it)
        try:
            self._fig.canvas.manager.set_window_title(title)  # type: ignore[union-attr]
        except AttributeError:
            pass

        self._world_x: float = float(world_x)
        self._world_y: float = float(world_y)
        self._road_network: Optional[RoadNetwork] = road_network
        self._step: int = 0

    def render(
        self,
        entities: List[Entity],
        sim_time: float = 0.0,
    ) -> None:
        """
        Redraw the entire scene for the current timestep.

        Parameters
        ----------
        entities : list[Entity]
            Entities to render.  Callers typically pass ``engine.entities.living()``
            but may include dead entities for post-mortem visualisation.
        sim_time : float
            Current simulation time in seconds (displayed in the title bar).
        """
        self._ax.cla()

        # ### Road network overlay (M2, NF-VIZ-006) ###
        if self._road_network is not None:
            self._draw_road_network()

        # ### Partition entities ###
        living: List[Entity] = [e for e in entities if e.alive]
        dead: List[Entity] = [e for e in entities if not e.alive]

        # ### Dead entities — grey translucent dots ###
        if dead:
            self._ax.scatter(
                [e.position[0] for e in dead],
                [e.position[1] for e in dead],
                c=_DEAD_COLOUR, s=14, alpha=0.30, zorder=2, label="_dead",
            )

        # Living entities — colour and shape by type (NF-VIZ-002)
        for entity in living:
            colour = _TYPE_COLOURS.get(entity.entity_type, "#607D8B")
            marker = _TYPE_MARKERS.get(entity.entity_type, "o")
            size = 100 if entity.is_eoi else 60
            edge_c = _EOI_EDGE_COLOUR if entity.is_eoi else "none"
            lw = 2.0 if entity.is_eoi else 0.0

            self._ax.scatter(
                entity.position[0], entity.position[1],
                c=colour, s=size, marker=marker,
                edgecolors=edge_c, linewidths=lw,
                zorder=4,
            )

            # Heading arrow proportional to XY speed (NF-VIZ-003)
            speed = float(np.linalg.norm(entity.velocity[:2]))
            if speed > 1e-6:
                arrow_len = min(speed * 2.0, self._world_x * 0.025)
                ux = entity.velocity[0] / speed
                uy = entity.velocity[1] / speed
                self._ax.annotate(
                    "",
                    xy=(entity.position[0] + ux * arrow_len,
                        entity.position[1] + uy * arrow_len),
                    xytext=(entity.position[0], entity.position[1]),
                    arrowprops=dict(
                        arrowstyle="->", color=colour,
                        lw=1.2, alpha=_HEADING_ALPHA,
                    ),
                    zorder=5,
                )

        # ### Legend ###
        legend_handles = [
            mpatches.Patch(color=c, label=t.name.replace("_", " ").title())
            for t, c in _TYPE_COLOURS.items()
        ]
        legend_handles += [
            mpatches.Patch(
                facecolor="white", edgecolor=_EOI_EDGE_COLOUR,
                linewidth=1.5, label="EOI (red ring)",
            ),
            mpatches.Patch(color=_DEAD_COLOUR, label="Dead"),
        ]
        if self._road_network is not None:
            legend_handles.append(
                mpatches.Patch(color=_ROAD_COLOUR, label="Road network")
            )
        self._ax.legend(
            handles=legend_handles, loc="upper right",
            fontsize=7, framealpha=0.85,
        )

        # ### Axes decoration ###
        self._ax.set_xlim(0.0, self._world_x)
        self._ax.set_ylim(0.0, self._world_y)
        self._ax.set_xlabel("East (m)")
        self._ax.set_ylabel("North (m)")
        self._ax.set_title(
            f"3DVES  t={sim_time:.1f}s  step={self._step}  "
            f"alive={len(living)}  dead={len(dead)}"
        )
        self._ax.grid(True, alpha=0.20, linestyle="--")
        self._ax.set_aspect("equal", adjustable="box")

        plt.draw()
        plt.pause(0.001)
        self._step += 1

    # ── Private helpers ───────────────────────────────────────────

    def _draw_road_network(self) -> None:
        """
        Draw road edges (grey lines) and nodes (small dots) (M2, NF-VIZ-006).

        Called before entities so roads appear behind entity markers.
        """
        rn = self._road_network
        if rn is None:
            return

        node_ids = rn.node_ids()

        # Draw edges by iterating adjacency structure
        drawn_pairs: set = set()
        for nid in node_ids:
            pos_a = rn.node_position(nid)
            # Access internal adjacency — acceptable here as viz is in same pkg
            for neighbour_id in rn._adjacency.get(nid, {}):
                pair = frozenset((nid, neighbour_id))
                if pair in drawn_pairs:
                    continue  # Skip reverse duplicate for bidirectional edges
                drawn_pairs.add(pair)
                pos_b = rn.node_position(neighbour_id)
                self._ax.plot(
                    [pos_a[0], pos_b[0]],
                    [pos_a[1], pos_b[1]],
                    color=_ROAD_COLOUR, linewidth=1.2,
                    alpha=0.60, zorder=1,
                )

        # Draw nodes as small dots
        if node_ids:
            xs = [rn.node_position(nid)[0] for nid in node_ids]
            ys = [rn.node_position(nid)[1] for nid in node_ids]
            self._ax.scatter(
                xs, ys, c=_NODE_COLOUR, s=12,
                zorder=2, alpha=0.70, marker="o",
            )
