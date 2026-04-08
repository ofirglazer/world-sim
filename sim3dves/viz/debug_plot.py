"""
sim3dves.viz.debug_plot
=======================
First-class 2-D top-down matplotlib simulation visualiser.

Reclassified in PRD v1.2 from "debug utility" to first-class product
deliverable.  Held to the same engineering standards as the simulation
engine (NF-VIZ-015): PEP8, full type annotations, NumPy docstrings,
dedicated unit tests.

M2 features
-----------
* Road network overlay (NF-VIZ-006 M2).
* Per-type marker shapes (NF-VIZ-002).

M3 features
-----------
* NFZ circles and geofence boundary (NF-VIZ-006 M3).
* Zoom via scroll-wheel, pan via right-click drag (NF-VIZ-008).
* Smooth pan: displacement computed from drag-start to prevent drift (NF-VIZ-016).
* Arrow-key pan proportional to current view extent (NF-VIZ-017).
* Window close detected; exposes ``window_closed`` property (NF-VIZ-018).
* Space key pauses/resumes simulation; ``paused`` property exposed (NF-VIZ-019).
* Reset view to default extent: "R" key and toolbar button (NF-VIZ-009).
* Entity selection by left-click; highlights selected marker (NF-VIZ-010).
* Inspection panel: FSM state, speed, destination; UAV extras: autopilot
  mode, endurance, low-fuel flag, NFZ violation flag, deconfliction role
  (NF-VIZ-011).
* Panel auto-updates every frame while entity is selected (NF-VIZ-012).
* Escape key or empty-space click deselects; panel hides (NF-VIZ-013).
* Panel is a fixed upper-left text overlay (NF-VIZ-014).

Visualiser roadmap (NF-VIZ-006)
--------------------------------
M1 : positions, heading arrows, EOI markers, legend, clock.
M2 : vehicle types, road network overlay.
M3 : NFZ circles, geofence, zoom/pan, entity inspection (this file).
M4 : FOV cone; rename DebugPlot -> SimulationView.
M5 : Track centroids and covariance ellipses.
M6 : Full road graph overlay.

NF-CE-001..005 compliant.  Implements: NF-VIZ-001..015.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from sim3dves.config.defaults import SimDefaults
from sim3dves.core.world import NFZCylinder
from sim3dves.entities.base import Entity, EntityType
from sim3dves.entities.uav import AutopilotMode
from sim3dves.maps.road_network import RoadNetwork

_D = SimDefaults()

# ### Module-level visual constants ###
_TYPE_COLOURS: Dict[EntityType, str] = {
    EntityType.PEDESTRIAN:      "#2196F3",   # Blue
    EntityType.WHEELED_VEHICLE: "#4CAF50",   # Green
    EntityType.TRACKED_VEHICLE: "#FF9800",   # Orange
    EntityType.UAV:             "#9C27B0",   # Purple
}

#  Marker shapes per type (NF-VIZ-002, M2)
_TYPE_MARKERS: Dict[EntityType, str] = {
    EntityType.PEDESTRIAN:      "o",  # Circle
    EntityType.WHEELED_VEHICLE: "s",  # Square
    EntityType.TRACKED_VEHICLE: "D",  # Diamond
    EntityType.UAV:             "^",  # Triangle
}

_EOI_EDGE_COLOUR: str = "#F44336"    # Red ring around EOI entities
_DEAD_COLOUR: str = "#BDBDBD"        # Grey for dead entities
_SELECTED_COLOUR: str = "#FFEB3B"    # Yellow highlight for selected entity
_ROAD_COLOUR: str = "#9E9E9E"        # Road edge colour
_NODE_COLOUR: str = "#757575"        # Road node colour
_NFZ_FILL_COLOUR: str = "#FF0000"    # NFZ fill (alpha applied separately)
_NFZ_EDGE_COLOUR: str = "#D32F2F"    # NFZ border colour
_GEOFENCE_COLOUR: str = "#FF6F00"    # Geofence boundary colour
_BACKGROUND_COLOUR: str = "#C0C0C0"  # Background classic silver color of the plot
_HEADING_ALPHA: float = 0.70         # Arrow transparency
_NFZ_ALPHA: float = 0.20             # NFZ fill transparency

# Hit-testing threshold in display pixels (NF-VIZ-010)
_SELECTION_THRESHOLD_PX: float = 15.0
# Zoom factor per scroll tick (NF-VIZ-008)
_ZOOM_FACTOR: float = 0.85
# Arrow-key pan step: computed each press as VIZ_PAN_KEY_STEP_FRAC * view width (NF-VIZ-017)


class DebugPlot:
    """
    Interactive top-down 2-D simulation visualiser (NF-VIZ-001..015).

    Designed to be called once per simulation step.  Axes are fully cleared
    and redrawn each frame; zoom/pan/selection state is preserved externally.

    Parameters
    ----------
    world_x : float
        World extent along East (X) axis in metres.
    world_y : float
        World extent along North (Y) axis in metres.
    road_network : RoadNetwork, optional
        Road edges and nodes drawn each frame (M2, NF-VIZ-006).
    nfz_cylinders : list[NFZCylinder], optional
        NFZ footprints drawn as circles (M3, NF-VIZ-006).
    title : str
        Window title.
    """

    def __init__(
        self,
        world_x: float,
        world_y: float,
        road_network: Optional[RoadNetwork] = None,
        nfz_cylinders: Optional[List[NFZCylinder]] = None,
        title: str = "Sim3Dves - Simulation View - phase M3 fixed",
    ) -> None:
        # Guard against re-enabling interactive mode on repeated construction
        if not plt.isinteractive():
            plt.ion()

        # Reserve 6 % of figure height at the bottom for the Reset button
        self._fig, self._ax = plt.subplots(figsize=(7.5, 7.5))
        self._ax.set_facecolor(_BACKGROUND_COLOUR)
        self._world_x: float = float(world_x)
        self._world_y: float = float(world_y)
        self._fig.subplots_adjust(bottom=0.07)

        try:
            self._fig.canvas.manager.set_window_title(title)  # type: ignore[union-attr]
        except AttributeError:
            pass

        self._world_x: float = float(world_x)
        self._world_y: float = float(world_y)
        self._road_network: Optional[RoadNetwork] = road_network
        self._nfz_cylinders: List[NFZCylinder] = nfz_cylinders or []
        self._step: int = 0

        # --- Zoom / pan state (NF-VIZ-008, NF-VIZ-016) ---
        self._xlim: Tuple[float, float] = (0.0, world_x)
        self._ylim: Tuple[float, float] = (0.0, world_y)
        self._is_panning: bool = False
        self._pan_start_display: Optional[Tuple[float, float]] = None
        # Stored at right-click press for smooth drift-free pan (NF-VIZ-016)
        self._xlim_at_pan_start: Tuple[float, float] = (0.0, world_x)
        self._ylim_at_pan_start: Tuple[float, float] = (0.0, world_y)

        # --- Selection state (NF-VIZ-010, NF-VIZ-013) ---
        self._selected_entity_id: Optional[str] = None
        # Snapshot of entities from the most recent render() call
        self._last_entities: List[Entity] = []

        # --- Window-close flag (NF-VIZ-018) ---
        self._window_closed: bool = False

        # --- Pause / resume state (NF-VIZ-019) ---
        self._paused: bool = False

        # --- Reset button (NF-VIZ-009) ---
        ax_btn = self._fig.add_axes([0.80, 0.01, 0.18, 0.04])
        self._btn_reset = Button(
            ax_btn, "Reset View [R]",
            color="#E3F2FD", hovercolor="#2196F3",
        )
        self._btn_reset.on_clicked(lambda _evt: self._reset_view())

        # --- Connect event handlers ---
        self._fig.canvas.mpl_connect("scroll_event",        self._on_scroll)
        self._fig.canvas.mpl_connect("button_press_event",  self._on_button_press)
        self._fig.canvas.mpl_connect("button_release_event", self._on_button_release)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event",     self._on_key)
        # NF-VIZ-018: window close terminates the simulation loop
        self._fig.canvas.mpl_connect("close_event",         self._on_close)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def render(
        self,
        entities: List[Entity],
        sim_time: float = 0.0,
    ) -> None:
        """
        Redraw the scene for the current timestep (NF-VIZ-001).

        Parameters
        ----------
        entities : list[Entity]
            Entities to render (typically ``engine.entities.living()``).
        sim_time : float
            Current simulation time in seconds (NF-VIZ-004).
        """
        self._last_entities = list(entities)
        self._ax.cla()

        # --- Static overlays (behind entities) ---
        self._draw_geofence()                           # NF-VIZ-006 M3
        if self._road_network is not None:
            self._draw_road_network()                   # NF-VIZ-006 M2
        if self._nfz_cylinders:
            self._draw_nfz_circles()                    # NF-VIZ-006 M3

        # --- Partition entities ---
        living: List[Entity] = [e for e in entities if e.alive]
        dead:   List[Entity] = [e for e in entities if not e.alive]

        # --- Dead entities: grey dots ---
        if dead:
            self._ax.scatter(
                [e.position[0] for e in dead],
                [e.position[1] for e in dead],
                c=_DEAD_COLOUR, s=14, alpha=0.30, zorder=2,
            )

        # --- Highlight selected entity (behind marker, NF-VIZ-010) ---
        sel = self._selected_entity
        if sel is not None and sel.alive:
            self._ax.scatter(
                sel.position[0], sel.position[1],
                s=350, marker="o",
                facecolors="none", edgecolors=_SELECTED_COLOUR,
                linewidths=3.0, zorder=6,
            )
            # --- Highlight selected entity destination and lines ---
            uav_waypoints = sel.entity_type == EntityType.UAV and sel.autopilot_mode == AutopilotMode.WAYPOINT
            uav_one_point = sel.entity_type == EntityType.UAV and (sel.autopilot_mode == AutopilotMode.ORBIT or sel.autopilot_mode == AutopilotMode.RTB)

            if (sel.entity_type == EntityType.TRACKED_VEHICLE and sel._waypoints) or uav_one_point:
                line_x = [sel.position[0]]
                line_y = [sel.position[1]]

                if uav_one_point:
                    dest_pos = sel.current_destination[0], sel.current_destination[1]
                else:
                    dest_pos = sel._waypoints[0][0], sel._waypoints[0][1]

                self._ax.scatter(
                    dest_pos[0], dest_pos[1],
                    s=350, marker="o",
                    facecolors="none", edgecolors=_SELECTED_COLOUR,
                    linewidths=3.0, zorder=6,
                )

                line_x.append(dest_pos[0])
                line_y.append(dest_pos[1])
                self._ax.plot(line_x, line_y, linestyle='dashed', color=_SELECTED_COLOUR)

            elif (sel.entity_type == EntityType.WHEELED_VEHICLE or uav_waypoints) and sel._waypoints:
                line_x = [sel.position[0]]
                line_y = [sel.position[1]]
                for idx in range(sel._waypoint_idx, len(sel._waypoints)):
                    next_x = sel._waypoints[idx][0]
                    next_y = sel._waypoints[idx][1]
                    self._ax.scatter(
                        next_x, next_y,
                        s=350, marker="o",
                        facecolors="none", edgecolors=_SELECTED_COLOUR,
                        linewidths=3.0, zorder=6,
                    )
                    line_x.append(next_x)
                    line_y.append(next_y)
                self._ax.plot(line_x, line_y, linestyle='dashed', color=_SELECTED_COLOUR)

        # Living entities — colour and shape by type (NF-VIZ-002)
        for entity in living:
            colour = _TYPE_COLOURS.get(entity.entity_type, "#607D8B")
            marker = _TYPE_MARKERS.get(entity.entity_type, "o")
            size = 120 if entity.is_eoi else 70
            edge_c = _EOI_EDGE_COLOUR if entity.is_eoi else "none"
            lw = 2.0 if entity.is_eoi else 0.0

            self._ax.scatter(
                entity.position[0], entity.position[1],
                c=colour, s=size, marker=marker,
                edgecolors=edge_c, linewidths=lw,
                zorder=5,
            )

            # Altitude annotation for UAVs (M3)
            if entity.entity_type == EntityType.UAV:
                self._ax.annotate(
                    f"{entity.position[2]:.0f}m",
                    xy=(entity.position[0], entity.position[1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color=colour, alpha=0.85, zorder=7,
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
                    zorder=6,
                )

        # --- Inspection panel (NF-VIZ-011, NF-VIZ-012, NF-VIZ-014) ---
        if sel is not None:
            self._draw_inspection_panel(sel)

        # --- Legend (NF-VIZ-005) ---
        self._draw_legend()

        # --- Axes decoration (NF-VIZ-004) ---
        self._ax.set_xlim(self._xlim)
        self._ax.set_ylim(self._ylim)
        self._ax.set_xlabel("East (m)")
        self._ax.set_ylabel("North (m)")
        # NF-VIZ-019: show pause state in title bar
        pause_str = "  [PAUSED]" if self._paused else ""
        self._ax.set_title(
            f"3DVES  t={sim_time:.1f}s  step={self._step}  "
            f"alive={len(living)}  dead={len(dead)}{pause_str}"
        )
        self._ax.grid(True, alpha=0.20, linestyle="--")
        self._ax.set_aspect("equal", adjustable="box")

        plt.draw()
        plt.pause(0.001)
        self._step += 1

    # -------------------------------------------------------------------------
    # Inspection panel helpers
    # -------------------------------------------------------------------------

    def _build_panel_text(self, entity: Entity) -> str:
        """
        Build the text content for the entity inspection panel (NF-VIZ-011).

        Accesses private members of VehicleEntity / UAVEntity using duck-typing
        with getattr/hasattr where public properties are not available.  This is
        intentional — the visualiser is an authorised internal consumer of
        simulation state (analogous to _adjacency access in _draw_road_network).

        Parameters
        ----------
        entity : Entity
            The currently selected entity.

        Returns
        -------
        str
            Multi-line text ready for display.
        """
        speed = float(np.linalg.norm(entity.velocity[:2]))
        lines = [
            f"ID      : {entity.entity_id[:16]}",
            f"Type    : {entity.entity_type.name}",
            f"State   : {entity.state.name}",
            f"Speed   : {speed:.1f} m/s",
            f"Heading : {entity.heading:.0f}",
            f"Pos     : ({entity.position[0]:.0f}, "
            f"{entity.position[1]:.0f}, {entity.position[2]:.0f})",
        ]

        # Destination — available on VehicleEntity and UAVEntity via property
        dest = getattr(entity, "current_destination", None)
        if dest is not None:
            lines.append(
                f"Dest   : ({dest[0]:.0f}, {dest[1]:.0f}, {dest[2]:.0f})"
            )

        # UAV-specific fields (NF-VIZ-011)
        if entity.entity_type == EntityType.UAV:
            mode = getattr(entity, "autopilot_mode", None)
            if mode is not None:
                lines.append(f"Mode   : {mode.name}")

            search_pattern = getattr(entity, "_search_pattern", None)
            if search_pattern is not None:
                lines.append(f"Pattern: {search_pattern.name}")

            endurance = getattr(entity, "endurance_remaining_s", None)
            if endurance is not None:
                lines.append(f"Endur  : {endurance:.0f}s")

            low_fuel = getattr(entity, "low_fuel", None)
            if low_fuel is not None:
                flag = "YES ⚠" if low_fuel else "no"
                lines.append(f"LowFuel: {flag}")

            nfz_viol = getattr(entity, "nfz_violated", None)
            if nfz_viol is not None:
                flag = "VIOLATION ✗" if nfz_viol else "clear"
                lines.append(f"NFZ    : {flag}")

            role = getattr(entity, "deconfliction_role", None)
            if role is not None:
                lines.append(f"Role   : {role}")

        return "\n".join(lines)

    def _draw_inspection_panel(self, entity: Entity) -> None:
        """
        Render the fixed upper-left inspection panel (NF-VIZ-011, NF-VIZ-014).

        Uses axes-fraction coordinates so the panel does not move when the
        user pans or zooms (NF-VIZ-012).

        Parameters
        ----------
        entity : Entity
            Currently selected entity.
        """
        text = self._build_panel_text(entity)
        self._ax.text(
            0.01, 0.99, text,
            transform=self._ax.transAxes,
            fontsize=7,
            family="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.82,
                edgecolor="#424242",
                linewidth=0.8,
            ),
            zorder=10,
        )

    # -------------------------------------------------------------------------
    # Event handlers (NF-VIZ-008, NF-VIZ-009, NF-VIZ-010, NF-VIZ-013)
    # -------------------------------------------------------------------------

    def _on_scroll(self, event: object) -> None:
        """
        Scroll-wheel zoom centred on cursor position (NF-VIZ-008).

        Zoom-in on scroll-up, zoom-out on scroll-down.  Limits preserved in
        self._xlim / self._ylim for use in the next render() frame.
        """
        if getattr(event, "inaxes", None) is not self._ax:
            return
        factor = (
            _ZOOM_FACTOR
            if getattr(event, "button", None) == "up"
            else 1.0 / _ZOOM_FACTOR
        )
        xdata: float = getattr(event, "xdata", 0.0) or 0.0
        ydata: float = getattr(event, "ydata", 0.0) or 0.0
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        new_xlim = (
            xdata + (xlim[0] - xdata) * factor,
            xdata + (xlim[1] - xdata) * factor,
        )
        new_ylim = (
            ydata + (ylim[0] - ydata) * factor,
            ydata + (ylim[1] - ydata) * factor,
        )
        self._xlim = new_xlim
        self._ylim = new_ylim
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._fig.canvas.draw_idle()

    def _on_button_press(self, event: object) -> None:
        """
        Left-click -> entity selection (NF-VIZ-010).
        Right-click -> start pan (NF-VIZ-008).
        """
        if getattr(event, "inaxes", None) is not self._ax:
            return
        button = getattr(event, "button", None)
        if button == 1:                     # Left-click: selection
            self._handle_entity_selection(event)
        elif button == 3:                   # Right-click: begin pan
            self._is_panning = True
            self._pan_start_display = (
                float(getattr(event, "x", 0.0)),
                float(getattr(event, "y", 0.0)),
            )
            # NF-VIZ-016: snapshot view limits at drag start so motion
            # events compute offset from origin — prevents drift
            self._xlim_at_pan_start = self._xlim
            self._ylim_at_pan_start = self._ylim

    def _on_button_release(self, event: object) -> None:
        """Release right-click: end pan (NF-VIZ-008)."""
        if getattr(event, "button", None) == 3:
            self._is_panning = False
            self._pan_start_display = None

    def _on_motion(self, event: object) -> None:
        """
        Right-click drag: pan the view (NF-VIZ-008).

        Uses display-pixel deltas to avoid coordinate-system feedback:
        as the axes limits change during pan, data coordinates for the same
        pixel shift; display coordinates do not.
        """
        if not self._is_panning:
            return
        if getattr(event, "inaxes", None) is not self._ax:
            return
        if self._pan_start_display is None:
            return

        ex: float = float(getattr(event, "x", 0.0))
        ey: float = float(getattr(event, "y", 0.0))
        # NF-VIZ-016: compute TOTAL delta from drag-start, not incremental delta.
        # This makes pan smooth and drift-free over long drags.
        dx_disp = ex - self._pan_start_display[0]
        dy_disp = ey - self._pan_start_display[1]

        ax_bbox = self._ax.get_window_extent()
        if ax_bbox.width == 0 or ax_bbox.height == 0:
            return

        # Convert pixel delta to data units using the FROZEN start limits
        # (not get_xlim()) so accumulated float error cannot build up.
        xlim0 = self._xlim_at_pan_start
        ylim0 = self._ylim_at_pan_start
        dx_data = -dx_disp * (xlim0[1] - xlim0[0]) / ax_bbox.width
        dy_data = -dy_disp * (ylim0[1] - ylim0[0]) / ax_bbox.height

        new_xlim = (xlim0[0] + dx_data, xlim0[1] + dx_data)
        new_ylim = (ylim0[0] + dy_data, ylim0[1] + dy_data)
        self._xlim = new_xlim
        self._ylim = new_ylim
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        # Immediate visual feedback without waiting for next sim step render
        self._fig.canvas.draw_idle()

    def _on_key(self, event: object) -> None:
        """
        Keyboard handler:
        * "r" / "R"     -> reset view (NF-VIZ-009).
        * "escape"      -> deselect entity (NF-VIZ-013).
        * arrow keys    -> pan proportional to view extent (NF-VIZ-017).
        * VIZ_PAUSE_KEY -> toggle pause/resume (NF-VIZ-019).
        """
        key = getattr(event, "key", "") or ""
        if key in ("r", "R", _D.VIZ_ZOOM_RESET_KEY):
            self._reset_view()
        elif key in ("escape", _D.VIZ_DESELECT_KEY):
            self._selected_entity_id = None    # NF-VIZ-013
        elif key == _D.VIZ_PAUSE_KEY:           # NF-VIZ-019
            self._paused = not self._paused
        elif key == "up":
            # NF-VIZ-017: step = VIZ_PAN_KEY_STEP_FRAC of current view height
            step = _D.VIZ_PAN_KEY_STEP_FRAC * (self._ylim[1] - self._ylim[0])
            new_ylim = (self._ylim[0] + step, self._ylim[1] + step)
            self._ylim = new_ylim
            self._ax.set_ylim(new_ylim)
            self._fig.canvas.draw_idle()
        elif key == "down":
            step = _D.VIZ_PAN_KEY_STEP_FRAC * (self._ylim[1] - self._ylim[0])
            new_ylim = (self._ylim[0] - step, self._ylim[1] - step)
            self._ylim = new_ylim
            self._ax.set_ylim(new_ylim)
            self._fig.canvas.draw_idle()
        elif key == "right":
            step = _D.VIZ_PAN_KEY_STEP_FRAC * (self._xlim[1] - self._xlim[0])
            new_xlim = (self._xlim[0] + step, self._xlim[1] + step)
            self._xlim = new_xlim
            self._ax.set_xlim(new_xlim)
            self._fig.canvas.draw_idle()
        elif key == "left":
            step = _D.VIZ_PAN_KEY_STEP_FRAC * (self._xlim[1] - self._xlim[0])
            new_xlim = (self._xlim[0] - step, self._xlim[1] - step)
            self._xlim = new_xlim
            self._ax.set_xlim(new_xlim)
            self._fig.canvas.draw_idle()

    def _reset_view(self) -> None:
        """
        Restore the viewport to the default full-world extent (NF-VIZ-009).

        Called by both the "R" key handler and the Reset toolbar button.
        """
        self._xlim = (0.0, self._world_x)
        self._ylim = (0.0, self._world_y)
        self._ax.set_xlim(self._xlim)
        self._ax.set_ylim(self._ylim)
        self._fig.canvas.draw_idle()

    def _on_close(self, event: object) -> None:
        """
        Handle matplotlib close_event: set ``_window_closed`` flag (NF-VIZ-018).

        The run loop polls ``plot.window_closed`` each step and breaks cleanly
        when this flag is True, allowing the logger context manager to flush
        and close the JSONL file before the process exits.
        """
        self._window_closed = True

    @property
    def window_closed(self) -> bool:
        """True after the visualiser window has been closed (NF-VIZ-018)."""
        return self._window_closed

    @property
    def paused(self) -> bool:
        """True when the simulation is paused via VIZ_PAUSE_KEY (NF-VIZ-019)."""
        return self._paused

    # -------------------------------------------------------------------------
    # Selection helpers
    # -------------------------------------------------------------------------

    def _handle_entity_selection(self, event: object) -> None:
        """
        Process a left-click event for entity selection (NF-VIZ-010, NF-VIZ-013).

        Converts the click to data coordinates and delegates to
        _find_nearest_entity.  If no entity is within the pixel threshold,
        the selection is cleared (NF-VIZ-013).

        Parameters
        ----------
        event : object
            matplotlib button_press_event.
        """
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            self._selected_entity_id = None
            return
        nearest = self._find_nearest_entity(float(xdata), float(ydata))
        self._selected_entity_id = (
            nearest.entity_id if nearest is not None else None
        )

    def _find_nearest_entity(
        self, x_data: float, y_data: float
    ) -> Optional[Entity]:
        """
        Return the living entity nearest to the clicked data-coordinate position.

        Hit-testing is performed in display-pixel space so the threshold is
        invariant under zoom (RSK-009).  Returns None when no entity is within
        _SELECTION_THRESHOLD_PX pixels of the click.

        Parameters
        ----------
        x_data : float
            Click X in data (world) coordinates.
        y_data : float
            Click Y in data (world) coordinates.

        Returns
        -------
        Entity or None
        """
        if not self._last_entities:
            return None

        # Convert click to display coordinates
        click_disp = self._ax.transData.transform([x_data, y_data])
        threshold_sq = _SELECTION_THRESHOLD_PX ** 2

        best: Optional[Entity] = None
        best_dist_sq: float = threshold_sq

        for entity in self._last_entities:
            if not entity.alive:
                continue
            ent_disp = self._ax.transData.transform(
                [entity.position[0], entity.position[1]]
            )
            dx = click_disp[0] - ent_disp[0]
            dy = click_disp[1] - ent_disp[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best = entity

        return best

    @property
    def _selected_entity(self) -> Optional[Entity]:
        """Return the currently selected entity object, or None (NF-VIZ-012)."""
        if self._selected_entity_id is None:
            return None
        return next(
            (e for e in self._last_entities
             if e.entity_id == self._selected_entity_id),
            None,
        )

    # -------------------------------------------------------------------------
    # Static draw helpers
    # -------------------------------------------------------------------------

    def _draw_legend(self) -> None:
        """Build and attach the legend (NF-VIZ-005)."""
        handles = [
            mpatches.Patch(color=c, label=t.name.replace("_", " ").title())
            for t, c in _TYPE_COLOURS.items()
        ]
        handles += [
            mpatches.Patch(
                facecolor="white", edgecolor=_EOI_EDGE_COLOUR,
                linewidth=1.5, label="EOI (red ring)",
            ),
            mpatches.Patch(color=_DEAD_COLOUR, label="Dead"),
        ]
        if self._selected_entity_id is not None:
            handles.append(
                mpatches.Patch(
                    facecolor="none", edgecolor=_SELECTED_COLOUR,
                    linewidth=2.0, label="Selected",
                )
            )
        if self._road_network is not None:
            handles.append(mpatches.Patch(color=_ROAD_COLOUR, label="Road network"))
        if self._nfz_cylinders:
            handles.append(
                mpatches.Patch(
                    facecolor=_NFZ_FILL_COLOUR, edgecolor=_NFZ_EDGE_COLOUR,
                    alpha=_NFZ_ALPHA, linewidth=1.5, label="NFZ",
                )
            )
        self._ax.legend(
            handles=handles, loc="upper right",
            fontsize=7, framealpha=0.85,
        )

    def _draw_geofence(self) -> None:
        """
        Draw world boundary and UAV geofence margin ring (M3, NF-VIZ-006).

        Outer dashed: hard world boundary (OOB kill line).
        Inner dotted: UAV_GEOFENCE_MARGIN_M trigger zone (FLR-005).
        """
        # Hard world boundary (dashed orange)
        boundary = mpatches.Rectangle(
            (0.0, 0.0), self._world_x, self._world_y,
            linewidth=1.5, edgecolor=_GEOFENCE_COLOUR,
            facecolor="none", linestyle="--", zorder=1, alpha=0.70,
        )
        self._ax.add_patch(boundary)

        # Geofence trigger margin (dotted inner rectangle)
        m = _D.UAV_GEOFENCE_MARGIN_M
        margin_rect = mpatches.Rectangle(
            (m, m), self._world_x - 2.0 * m, self._world_y - 2.0 * m,
            linewidth=1.0, edgecolor=_GEOFENCE_COLOUR,
            facecolor="none", linestyle=":", zorder=1, alpha=0.40,
        )
        self._ax.add_patch(margin_rect)

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
            # Access internal adjacency - acceptable here as viz is in same pkg
            for neighbour_id in rn._adjacency.get(nid, {}):
                pair = frozenset((nid, neighbour_id))
                if pair in drawn_pairs:
                    continue  # Skip reverse duplicate for bidirectional edges
                drawn_pairs.add(pair)
                pos_b = rn.node_position(neighbour_id)
                self._ax.plot(
                    [pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
                    color=_ROAD_COLOUR, linewidth=1.2, alpha=0.60, zorder=1,
                )

        # Draw nodes as small dots
        if node_ids:
            xs = [rn.node_position(nid)[0] for nid in node_ids]
            ys = [rn.node_position(nid)[1] for nid in node_ids]
            self._ax.scatter(
                xs, ys, c=_NODE_COLOUR, s=12, zorder=2, alpha=0.70, marker="o",
            )

    def _draw_nfz_circles(self) -> None:
        """
        Draw NFZ cylinder footprints as translucent red circles (M3, NF-VIZ-006).

        Each circle represents the horizontal XY extent of one NFZCylinder.
        The altitude ceiling is annotated at the centre.
        """
        for nfz in self._nfz_cylinders:
            circle = mpatches.Circle(
                (float(nfz.center_xy[0]), float(nfz.center_xy[1])),
                radius=float(nfz.radius_m),
                facecolor=_NFZ_FILL_COLOUR,
                edgecolor=_NFZ_EDGE_COLOUR,
                linewidth=1.5,
                alpha=_NFZ_ALPHA,
                zorder=3,
            )
            self._ax.add_patch(circle)
            self._ax.text(
                float(nfz.center_xy[0]),
                float(nfz.center_xy[1]),
                f"NFZ\n{chr(0x2264)}{nfz.alt_max_m:.0f}m",
                ha="center", va="center",
                fontsize=6, color=_NFZ_EDGE_COLOUR,
                fontweight="bold", zorder=4, alpha=0.85,
            )
