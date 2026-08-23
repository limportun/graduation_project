"""Physical constants and configuration for the MagRobotNav environment."""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

@dataclass(frozen=True)
class EnvConfig:
    """Centralized physical constants for robot navigation simulation.

    All length units are in millimeters, angles in radians, time in seconds.
    """

    # --- Unit conversion ---
    mm_per_pixel: float = 0.0443478093203021
    v_per_i: float = 2.0  # mm/s per A (motor constant)

    # --- Image dimensions (pixels) ---
    image_width: int = 1920
    image_height: int = 1080

    max_obstacles_num: int = 10

    # --- Time step ---
    dt: float = 0.2

    # --- Robot footprint (millimeters) ---
    robot_length: float = 6.0
    robot_width: float = 3.0

    # --- Safety & arrival threshold (millimeters) ---
    safe_dist: float = 0.2
    arrival_threshold: float = 1.5

    # --- Robot velocity limits ---
    min_velocity: float = 0.0
    max_velocity: float = 2.0
    max_acceleration: float = 2.0
    min_angular_velocity: float = field(default_factory=lambda: -np.deg2rad(45))
    max_angular_velocity: float = field(default_factory=lambda: np.deg2rad(45))
    max_angular_acceleration: float = field(default_factory=lambda: np.deg2rad(90))

    # --- Obstacle velocity limits ---
    min_obstacle_angular_velocity: float = field(default_factory=lambda: -np.deg2rad(90))
    max_obstacle_angular_velocity: float = field(default_factory=lambda: np.deg2rad(90))
    max_obstacle_angular_acceleration: float = field(default_factory=lambda: np.deg2rad(180))
    min_obstacle_velocity: float = -0.5
    max_obstacle_velocity: float = 0.5
    max_obstacle_acceleration: float = 1.0

    # --- Computed properties ---
    @property
    def robot_radius(self) -> float:
        """Half-diagonal of the robot footprint."""
        return float(np.hypot(self.robot_length / 2, self.robot_width / 2))

    @property
    def roi_length_x(self) -> float:
        """Region-of-interest width in millimeters."""
        return self.image_width * self.mm_per_pixel

    @property
    def roi_length_y(self) -> float:
        """Region-of-interest height in millimeters."""
        return self.image_height * self.mm_per_pixel

    @property
    def max_obstacle_radius(self) -> float:
        return self.roi_length_x * 0.015

    @property
    def min_obstacle_radius(self) -> float:
        return self.roi_length_x * 0.005


# Default config instance
_default = EnvConfig()

# --- Backward-compatible module-level constants ---
MM_PER_PIXEL = _default.mm_per_pixel
V_PER_I = _default.v_per_i
IMAGE_WIDTH = _default.image_width
IMAGE_HEIGHT = _default.image_height
MAX_OBSTACLES_NUM = _default.max_obstacles_num
DT = _default.dt
ROBOT_LENGTH = _default.robot_length
ROBOT_WIDTH = _default.robot_width
ROBOT_RADIUS = _default.robot_radius
SAFE_DIST = _default.safe_dist
ARRIVAL_THRESHOLD = _default.arrival_threshold
ROI_LENGTH_X = _default.roi_length_x
ROI_LENGTH_Y = _default.roi_length_y
MAX_OBSTACLE_RADIUS = _default.max_obstacle_radius
MIN_OBSTACLE_RADIUS = _default.min_obstacle_radius
MIN_VELOCITY = _default.min_velocity
MAX_VELOCITY = _default.max_velocity
MAX_ACCELERATION = _default.max_acceleration
MIN_ANGULAR_VELOCITY = _default.min_angular_velocity
MAX_ANGULAR_VELOCITY = _default.max_angular_velocity
MAX_ANGULAR_ACCELERATION = _default.max_angular_acceleration
MIN_OBSTACLE_ANGULAR_VELOCITY = _default.min_obstacle_angular_velocity
MAX_OBSTACLE_ANGULAR_VELOCITY = _default.max_obstacle_angular_velocity
MAX_OBSTACLE_ANGULAR_ACCELERATION = _default.max_obstacle_angular_acceleration
MIN_OBSTACLE_VELOCITY = _default.min_obstacle_velocity
MAX_OBSTACLE_VELOCITY = _default.max_obstacle_velocity
MAX_OBSTACLE_ACCELERATION = _default.max_obstacle_acceleration

# --- Obstacle shapes ---
class ObstacleShape(Enum):
    circle = auto()
    polygon = auto()


# --- Environment types ---
class EnvType(Enum):
    circle = auto()
    square = auto()
    vessel = auto()
    maze = auto()


# --- PRM sub-goal planning states ---
class PlanningState(Enum):
    DIRECT = auto()      # walk straight to goal, no planning needed
    FOLLOWING = auto()   # following a PRM-planned path via sub-goals