"""
sim3dves.config.defaults
========================
Single source of truth for all configurable scalar constants.

NF-M-006: All tunable parameters are externalized here.
NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SimDefaults:
    """
    Immutable bag of simulation-wide default values.

    Using a frozen dataclass (rather than module-level constants)
    gives us a single injectable object, easy to override in tests,
    and discoverable via IDE auto-complete.
    """

    # ### Pedestrian (PED-001) ###
    PED_SPEED_MIN_MPS: float = 1.0  # Minimum pedestrian speed (m/s)
    PED_SPEED_MAX_MPS: float = 1.8  # Maximum pedestrian speed (m/s)
    PED_VELOCITY_NOISE_STD: float = 0.05  # Heading-angle noise σ (radians/step)
    PED_NUMBER: int = 40

    # ### World (ENV-001) ###
    WORLD_EXTENT_X_M: float = 200.0   # Default world width  (m)
    WORLD_EXTENT_Y_M: float = 200.0   # Default world height (m)
    WORLD_ALT_FLOOR_M: float = 0.0     # Minimum AGL altitude (m)
    WORLD_ALT_CEIL_M: float = 500.0    # Maximum AGL altitude (m)

    # ### Simulation clock (SIM-001) ###
    SIM_DT_S: float = 0.1              # Default timestep (seconds)
    SIM_DURATION_S: float = 60.0       # Default scenario duration (seconds)
    SIM_SEED: int = 42                 # Default RNG seed (SIM-003)

    # ### Logger (LOG-001) ###
    LOG_FILE: str = "sim_log.jsonl"    # Default JSONL output path
