"""
sim3dves.web.serialiser
=======================
Frame serialiser: converts one simulation step's state into a
JSON-serialisable dict that the WebSocket broadcaster sends to every
connected browser client.

This module is the web-layer analogue of ``DebugPlot.render()``.
It has zero coupling to FastAPI, threading, or asyncio — it is a pure
function of the engine's public state after each ``engine.step()`` call.

World coordinates are sent verbatim in metres.  The browser owns the
world-to-canvas transform via its local ``viewState`` object so every
client can zoom and pan independently without the server knowing.

The 2×2 position covariance block is included verbatim for each track so
the browser can eigen-decompose it (six lines of JavaScript) to draw a
properly oriented error ellipse without any server-side linear algebra.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from sim3dves.entities.base import Entity, EntityType
from sim3dves.payload.track_manager import TrackManager


def serialise_frame(
    entities: List[Entity],
    sim_time: float,
    step_idx: int,
    detected_ids: Set[str],
    track_manager: Optional[TrackManager] = None,
) -> Dict[str, Any]:
    """
    Produce a JSON-serialisable snapshot of one simulation step.

    Parameters
    ----------
    entities : list[Entity]
        Living entities this step -- engine.entities.living().
    sim_time : float
        Current simulation time (s) -- engine.sim_time.
    step_idx : int
        Zero-based step counter -- engine.step_idx.
    detected_ids : set[str]
        Entity IDs detected this step -- engine.step_detections.
    track_manager : TrackManager, optional
        Active Kalman-filter track registry.

    Returns
    -------
    dict
        Wire frame ready for json.dumps().
        Keys: t, step, entities[], tracks[]
    """
    entity_frames: List[Dict[str, Any]] = []

    for ent in entities:
        rec: Dict[str, Any] = {
            "id":       ent.entity_id,
            "type":     ent.entity_type.name,
            "pos":      [round(float(v), 2) for v in ent.position.tolist()],
            "vel":      [round(float(v), 3) for v in ent.velocity.tolist()],
            "heading":  round(float(ent.heading), 1),
            "is_eoi":   ent.is_eoi,
            "state":    ent.state.name,
            "detected": ent.entity_id in detected_ids,
        }

        if ent.entity_type is EntityType.UAV:
            ap_mode = getattr(ent, "_autopilot_mode", None)
            rec["autopilot_mode"]  = ap_mode.name if ap_mode is not None else None
            rec["endurance_s"]     = round(float(getattr(ent, "_endurance_remaining_s", 0.0)), 1)
            rec["low_fuel"]        = bool(getattr(ent, "low_fuel", False))
            rec["nfz_violated"]    = bool(getattr(ent, "_nfz_violated_flag", False))
            rec["deconflict_role"] = str(getattr(ent, "_deconfliction_role", "PRIMARY"))

            payload = getattr(ent, "payload", None)
            if payload is not None:
                tip  = getattr(payload, "fov_tip_world", None)
                axis = getattr(payload, "fov_axis_world", None)
                mode = getattr(payload, "mode", None)
                rec["fov"] = {
                    "tip":        tip.tolist()  if tip  is not None else None,
                    "axis":       axis.tolist() if axis is not None else None,
                    "half_angle": float(getattr(payload, "fov_half_angle_deg", 10.0)),
                    "gimbal_az":  round(float(getattr(payload, "gimbal_az_deg", 0.0)), 1),
                    "gimbal_el":  round(float(getattr(payload, "gimbal_el_deg", -45.0)), 1),
                    "mode":       mode.name if mode is not None else None,
                }
            else:
                rec["fov"] = None

        entity_frames.append(rec)

    tracks: List[Dict[str, Any]] = []
    if track_manager is not None:
        for eid, trk in track_manager.active_tracks.items():
            cov = trk.position_covariance
            tracks.append({
                "entity_id": eid,
                "quality":   trk.quality.name,
                "pos_xy":    [round(float(v), 2) for v in trk.position_xy.tolist()],
                "cov": [
                    [round(float(cov[0, 0]), 3), round(float(cov[0, 1]), 3)],
                    [round(float(cov[1, 0]), 3), round(float(cov[1, 1]), 3)],
                ],
                "is_eoi":    trk.is_eoi,
                "age_steps": trk.age_steps,
            })

    return {
        "t":        round(float(sim_time), 2),
        "step":     step_idx,
        "entities": entity_frames,
        "tracks":   tracks,
    }
