"""
sim3dves.viz.debug_plot
=======================
Real-time 2-D top-down matplotlib debug visualiser.

Provides a minimal M1 baseline view that will be extended incrementally
as subsequent milestones add UAV tracks, payload FOV cones, road networks,
NFZ overlays, and track quality heat-maps.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-005: Inline comments throughout.
Implements: M1 baseline visualisation requirement.

FIX vs original:
----------------
1. No entity-type color differentiation — all entities rendered identically.
2. No heading arrows — direction of travel invisible.
3. No EOI visual indicator.
4. No dead-entity rendering (dead entities simply not filtered, could crash
   on stale references).
5. No sim-time / step counter in title.
6. No legend.
7. ``plt.ion()`` inside the constructor causes issues if called multiple
   times — guard added.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sim3dves.entities.base import Entity, EntityType

# ### Colour palette ###
_TYPE_COLOURS: Dict[EntityType, str] = {
    EntityType.PEDESTRIAN:      "#2196F3",   # Blue
    EntityType.WHEELED_VEHICLE: "#4CAF50",   # Green
    EntityType.TRACKED_VEHICLE: "#FF9800",   # Orange
    EntityType.UAV:             "#9C27B0",   # Purple
}
_EOI_EDGE_COLOUR: str = "#F44336"    # Red ring around EOI entities
_DEAD_COLOUR: str = "#BDBDBD"        # Gray for dead entities
_HEADING_ALPHA: float = 0.7          # Arrow transparency


class DebugPlot:
    """
    Interactive top-down 2-D visualiser backed by matplotlib.

    Designed to be called once per simulation step inside the run loop.
    Rendering is intentionally lightweight — the axes are fully cleared
    (``cla()``) and redrawn each frame.  For smoother animation in later
    milestones, consider blitting or ``FuncAnimation``.

    Incremental feature roadmap
    ---------------------------
    M1  : Pedestrian positions, headings, EOI markers, legend, clock.
    M2  : UAV positions and altitude annotation.
    M3  : NFZ circles, geofence boundary.
    M4  : Payload FOV cone projected onto ground plane.
    M5  : Active track centroids and covariance ellipses.
    M6  : Road network graph overlay.
    """

    def __init__(
        self,
        world_x: float,
        world_y: float,
        title: str = "3DVES — Debug View",
    ) -> None:
        """
        Parameters
        ----------
        world_x : float
            World extent along the East (X) axis in metres.
        world_y : float
            World extent along the North (Y) axis in metres.
        title : str
            Window title string.
        """
        # Guard against re-enabling interactive mode on repeated construction
        if not plt.isinteractive():
            plt.ion()

        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        # Set window title safely (backend may not support it)
        try:
            self._fig.canvas.manager.set_window_title(title)  # type: ignore[union-attr]
        except AttributeError:
            pass

        self._world_x: float = world_x
        self._world_y: float = world_y
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

        living: List[Entity] = [e for e in entities if e.alive]
        dead: List[Entity] = [e for e in entities if not e.alive]

        # ### Dead entities — grey translucent dots ###
        if dead:
            xs = [e.position[0] for e in dead]
            ys = [e.position[1] for e in dead]
            self._ax.scatter(xs, ys, c=_DEAD_COLOUR, s=18, alpha=0.35, zorder=1,
                             label="Dead")

        # ### Living entities — colored by type ###
        for entity in living:
            colour = _TYPE_COLOURS.get(entity.entity_type, "#607D8B")
            # UAVs rendered as triangles to suggest aerial nature
            marker = "^" if entity.entity_type == EntityType.UAV else "o"
            size = 90 if entity.is_eoi else 55
            edge_colour = _EOI_EDGE_COLOUR if entity.is_eoi else "none"
            linewidth = 1.8 if entity.is_eoi else 0.0

            self._ax.scatter(
                entity.position[0],
                entity.position[1],
                c=colour,
                s=size,
                marker=marker,
                edgecolors=edge_colour,
                linewidths=linewidth,
                zorder=2,
            )

            # ### Heading arrow — proportional to XY speed ###
            speed = float(np.linalg.norm(entity.velocity[:2]))
            if speed > 1e-6:
                # Arrow length scales with speed but is capped at 3% of world
                arrow_len = min(speed * 2.5, self._world_x * 0.03)
                ux = entity.velocity[0] / speed   # Unit vector X
                uy = entity.velocity[1] / speed   # Unit vector Y
                self._ax.annotate(
                    "",
                    xy=(entity.position[0] + ux * arrow_len,
                        entity.position[1] + uy * arrow_len),
                    xytext=(entity.position[0], entity.position[1]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=colour,
                        lw=1.2,
                        alpha=_HEADING_ALPHA,
                    ),
                    zorder=3,
                )

        # ### Legend ###
        legend_handles = [
            mpatches.Patch(color=c, label=t.name.replace("_", " ").title())
            for t, c in _TYPE_COLOURS.items()
        ]
        legend_handles += [
            mpatches.Patch(
                facecolor="white",
                edgecolor=_EOI_EDGE_COLOUR,
                linewidth=1.5,
                label="EOI (red ring)",
            ),
            mpatches.Patch(color=_DEAD_COLOUR, label="Dead"),
        ]
        self._ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                        framealpha=0.85)

        # ── Axes decoration ────────────────────────────────────────
        self._ax.set_xlim(0.0, self._world_x)
        self._ax.set_ylim(0.0, self._world_y)
        self._ax.set_xlabel("East (m)")
        self._ax.set_ylabel("North (m)")
        self._ax.set_title(
            f"3DVES  t={sim_time:.1f}s  step={self._step}  "
            f"alive={len(living)}  dead={len(dead)}"
        )
        self._ax.grid(True, alpha=0.25, linestyle="--")
        self._ax.set_aspect("equal", adjustable="box")

        plt.draw()
        plt.pause(0.001)
        self._step += 1
