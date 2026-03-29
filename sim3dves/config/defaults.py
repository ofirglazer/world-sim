"""
sim3dves.config.defaults
========================
Single source of truth for all configurable scalar constants.

NF-M-006: All tunable parameters are externalized here.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.

M2 additions: vehicle kinematics, social-force parameters,
road-network speed limit, EntityManager neighbor-search radius.
"""
from dataclasses import dataclass


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
    PED_NUMBER: int = 40

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

    # ### EntityManager context (NF-P-001) ###
    NEIGHBOR_RADIUS_M: float = 10.0  # Radius for social-force neighbor search

    # ### World (ENV-001) ###
    WORLD_EXTENT_X_M: float = 200.0   # Default world width  (m)
    WORLD_EXTENT_Y_M: float = 200.0   # Default world height (m)
    WORLD_ALT_FLOOR_M: float = 0.0     # Minimum AGL altitude (m)
    WORLD_ALT_CEIL_M: float = 500.0    # Maximum AGL altitude (m)

    # ### Simulation clock (SIM-001) ###
    SIM_DT_S: float = 0.1              # Default timestep (seconds)
    SIM_DURATION_S: float = 20.0       # Default scenario duration (seconds)
    SIM_SEED: int = 42                 # Default RNG seed (SIM-003)

    # ### Logger (LOG-001) ###
    LOG_FILE: str = "sim_log.jsonl"    # Default JSONL output path
