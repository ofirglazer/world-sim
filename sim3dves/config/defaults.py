"""
sim3dves.config.defaults
========================
Single source of truth for all configurable scalar constants.

NF-M-006: All tunable parameters are externalized here.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.

M2 additions: vehicle kinematics, social-force parameters,
  road-network speed limit, EntityManager neighbor-search radius.
M3 additions: UAV kinematics, endurance, flight rules, search pattern
              parameters, wind model, deconfliction constants.
M3 fixes    : Added UAV_CORNER_ESCAPE_MARGIN_M and UAV_SEARCH_MARGIN_M
              (FLR-008, FLR-011) to deconflict search patterns from geofence.
              World defaults (WORLD_ALT_FLOOR_M, WORLD_ALT_CEIL_M) are now
              consumed by World.__init__ so they are no longer unused.
"""
from dataclasses import dataclass
import numpy


@dataclass(frozen=True)
class SimDefaults:
    """
    Immutable container of simulation-wide defaults (NF-M-006).

    Using a frozen dataclass (rather than module-level constants)
    gives us a single injectable object, easy to override in tests,
    and discoverable via IDE auto-complete.
    """

    # ### Pedestrian (PED-001) ###
    PED_SPEED_MIN_MPS: float = 1.0  # Minimum pedestrian speed (m/s)
    PED_SPEED_MAX_MPS: float = 1.8  # Maximum pedestrian speed (m/s)
    PED_VELOCITY_NOISE_STD: float = 0.05  # Angular heading noise σ (radians/step)

    # ### Social Force (PED-003) ###
    SOCIAL_RADIUS_M: float = 5.0  # Repulsion cutoff distance (m)
    SOCIAL_REPULSION_K: float = 1.5  # Repulsion strength (m/s per unit)

    # ### Vehicle kinematics (VEH-002) ###
    VEH_MAX_SPEED_MPS: float = 12.0  # Wheeled vehicle top speed (~43 km/h)
    VEH_ACCEL_MPS2: float = 2.0  # Acceleration (m/s²)
    VEH_DECEL_MPS2: float = 4.0  # Deceleration (m/s²)
    VEH_MAX_TURN_RATE_DPS: float = 60.0  # Max yaw rate (deg/s)
    VEH_ARRIVAL_THRESHOLD_M: float = 4.0  # Waypoint arrival radius (m)

    # ### Tracked vehicle (VEH-001) ###
    VEH_TRACKED_MAX_SPEED_MPS: float = 8.0  # Tracked top speed (m/s)
    VEH_OFF_ROAD_FACTOR: float = 0.50  # Speed factor when off-road

    # ### Road network (ENV-006) ###
    ROAD_SPEED_LIMIT_MPS: float = 13.9  # Default edge speed limit (~50 km/h)
    GRID_ROWS: int = 6
    GRID_COLS: int = 6
    GRID_SPACING_M: float = 100.0
    GRID_ORIGIN = numpy.array([50.0, 50.0])

    # ### EntityManager neighbor search (NF-P-001) ###
    # Used for pedestrian social-force context.  Ground entities ignore
    # neighbors outside SOCIAL_RADIUS_M anyway, so increasing this is safe.
    NEIGHBOR_RADIUS_M: float = 10.0  # Ground entity search radius (m)

    # ### World (ENV-001) ###
    WORLD_EXTENT_X_M: float = 600.0  # Default world width  (m)
    WORLD_EXTENT_Y_M: float = 600.0  # Default world height (m)
    WORLD_ALT_FLOOR_M: float = 0.0  # Minimum AGL altitude (m)
    WORLD_ALT_CEIL_M: float = 500.0  # Maximum AGL altitude (m)

    # ### Simulation clock (SIM-001) ###
    SIM_DT_S: float = 0.1  # Default timestep (seconds)
    SIM_DURATION_S: float = 60.0  # Default scenario duration (seconds)
    SIM_SEED: int = 42  # Default RNG seed (SIM-003)

    # ### Populating sim ###
    NUM_WHEELED: int = 12
    NUM_TRACKED: int = 5
    NUM_PEDESTRIANS: int = 40
    NUM_UAVS: int = 4

    # ### Logger (LOG-001) ###
    LOG_FILE: str = "sim_log.jsonl"  # Default JSONL output path

    # ==========================================================================
    # UAV subsystem
    # ==========================================================================

    # ### UAV platform kinematics (UAV-001, UAV-002) ###
    UAV_MAX_SPEED_MPS: float = 25.0  # Max horizontal speed (m/s, ~90 km/h)
    UAV_CLIMB_RATE_MPS: float = 5.0  # Max climb rate (m/s)
    UAV_DESCENT_RATE_MPS: float = 3.0  # Max descent rate (m/s)
    UAV_TURN_RATE_DPS: float = 30.0  # Max yaw rate (deg/s) — UAV-002
    UAV_CRUISE_ALT_M: float = 100.0  # Default cruise altitude AGL (m)
    UAV_ALT_FLOOR_M: float = 30.0  # Min operating altitude AGL (FLR-002)
    UAV_ALT_CEIL_M: float = 500.0  # Max operating altitude AGL (FLR-003)
    UAV_ARRIVAL_THRESHOLD_M: float = 15.0  # Waypoint arrival radius (m)

    # ### UAV endurance (UAV-003) ###
    UAV_ENDURANCE_S: float = 600.0  # Default endurance budget (s)
    UAV_LOW_FUEL_THRESHOLD_S: float = 90.0  # Low-fuel alert threshold (FLR-006)

    # ### UAV flight rules ###
    UAV_SEPARATION_M: float = 50.0  # Min inter-UAV separation (FLR-004)
    UAV_NFZ_LOOKAHEAD_S: float = 10.0  # NFZ predictive check horizon (FLR-001)
    UAV_GEOFENCE_MARGIN_M: float = 60.0  # Distance from boundary → avoidance (FLR-005)
    UAV_ORBIT_RADIUS_M: float = 100.0  # Default orbit/cued-slew radius (FLR-009)
    UAV_ORBIT_SPEED_MPS: float = 15.0  # Tangential orbit speed (m/s)
    UAV_LOITER_RADIUS_M: float = 50.0  # Holding-pattern orbit radius (m)
    UAV_LOITER_SPEED_MPS: float = 10.0  # Loiter tangential speed (m/s)

    # ### Corner-escape geometry (FLR-011) [NEW in M3 fix] ###
    # At max_speed=25 m/s and turn_rate=30 deg/s the minimum turn radius is
    # 25 / (30*pi/180) ~= 48 m.  A 135-degree corner escape arc travels
    # (135/360)*2*pi*48 ~= 113 m.  UAV_CORNER_ESCAPE_MARGIN_M is rounded up
    # to 120 m to add a small safety buffer.
    UAV_CORNER_ESCAPE_MARGIN_M: float = 120.0  # Extra inner margin for corner turns (m)

    # ### Search pattern safe boundary (FLR-008) [NEW in M3 fix] ###
    # UAV_SEARCH_MARGIN_M = UAV_GEOFENCE_MARGIN_M + UAV_CORNER_ESCAPE_MARGIN_M
    # All pattern waypoints must stay >= this distance from every world edge so
    # the UAV can never trigger the geofence RTB rule while executing a normal
    # search leg or end-of-strip turn.
    UAV_SEARCH_MARGIN_M: float = 180.0         # = 60 + 120 (m)

    # ### Multi-UAV deconfliction (FLR-010) ###
    UAV_DECONFLICTION_ALT_STEP_M: float = 30.0  # SECONDARY altitude offset (m)
    UAV_SAME_ORBIT_THRESHOLD_M: float = 30.0  # Cue points this close → same orbit

    # ### Wind model (FLR-007) ###
    UAV_WIND_X_MPS: float = 0.0  # Wind East component (m/s)
    UAV_WIND_Y_MPS: float = 0.0  # Wind North component (m/s)
    UAV_WIND_Z_MPS: float = 0.0  # Wind Up component (m/s)

    # ### Search pattern parameters (FLR-008) ###
    UAV_LAWNMOWER_STRIP_W_M: float = 150.0  # Lawnmower strip width (m)
    UAV_SPIRAL_STRIP_W_M: float = 100.0  # Expanding-spiral radial increment (m)
    UAV_RANDOM_WALK_WAYPOINTS: int = 20  # Number of random-walk waypoints

    # ### EntityManager (M3: per-entity neighbor radius) ###
    # UAVs need a larger search radius for FLR-004 separation detection.
    UAV_NEIGHBOR_RADIUS_M: float = 200.0  # UAV neighbor search radius (m)

    # ### UAV NFZ ###
    NFZ_DEFINITIONS = [
        # (center_x, center_y, radius_m, alt_max_m)
        (200.0, 300.0, 60.0, 200.0),
        (450.0, 150.0, 50.0, 150.0),
        (400.0, 450.0, 70.0, 300.0),
    ]

    # ### Visualiser interactive controls (NF-VIZ-008-015) ###
    VIZ_ZOOM_RESET_KEY: str = "r"             # Keyboard shortcut to reset view
    VIZ_DESELECT_KEY: str = "escape"          # Keyboard shortcut to deselect entity
    VIZ_HIT_THRESHOLD_FRAC: float = 0.025     # Click hit-test radius as fraction of view width
    VIZ_DRAG_THRESHOLD_PX: float = 5.0        # Pixel movement to distinguish drag from click
    VIZ_PANEL_ALPHA: float = 0.80             # Inspection panel background transparency
