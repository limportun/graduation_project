"""Backward-compatibility re-export shim.

All functionality has been moved to sub-modules:
  - geometry.py   : pi_2_pi, euclidean_distance, point_to_line_distance,
                    transform_to_robot_frame
  - models.py     : Command, Robot, Obstacle, Goal
  - utils.py      : load_config, generate_convex_polygon,
                    transform_vertices_to_world, generate_random_obstacles
  - laser.py      : calculate_laser_scan_optimized, compute_rectangle_robot_scan
  - rendering.py  : convert_obstacles_to_image_RGB
  - dwa_config.py : Config, GoalConfig

Prefer importing from the specific sub-module in new code.
"""

from .config import EnvType  # re-export for backward compatibility
from .geometry import (
    pi_2_pi,
    euclidean_distance,
    point_to_line_distance,
    transform_to_robot_frame,
)
from .models import Command, Robot, Obstacle, Goal
from .utils import (
    load_config,
    generate_convex_polygon,
    transform_vertices_to_world,
    generate_random_obstacles,
)
from .laser import (
    calculate_laser_scan_optimized,
    compute_rectangle_robot_scan,
)
from .rendering import convert_obstacles_to_image_RGB
from .dwa_config import Config, GoalConfig

__all__ = [
    # config
    "EnvType",
    # geometry
    "pi_2_pi", "euclidean_distance", "point_to_line_distance",
    "transform_to_robot_frame",
    # models
    "Command", "Robot", "Obstacle", "Goal",
    # utils
    "load_config", "generate_convex_polygon", "transform_vertices_to_world",
    "generate_random_obstacles",
    # laser
    "calculate_laser_scan_optimized", "compute_rectangle_robot_scan",
    # rendering
    "convert_obstacles_to_image_RGB",
    # dwa_config
    "Config", "GoalConfig",
]
