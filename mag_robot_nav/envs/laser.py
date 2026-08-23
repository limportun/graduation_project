"""Vectorized LiDAR and robot-footprint scan utilities."""

from typing import Iterable, List

import numpy as np
from shapely.geometry import LineString, Polygon

from .config import ObstacleShape
from .models import Obstacle
from .utils import transform_vertices_to_world


def _cross_2d(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _update_from_segments(scan, directions, origin, range_max, starts, ends):
    """Update ray distances using vectorized ray/segment intersections."""
    if len(starts) == 0:
        return
    starts = np.asarray(starts, dtype=float)
    segments = np.asarray(ends, dtype=float) - starts
    relative_starts = starts - origin

    ray = directions[:, None, :]
    segment = segments[None, :, :]
    denominator = _cross_2d(ray, segment)
    non_parallel = np.abs(denominator) > 1e-12
    safe_denominator = np.where(non_parallel, denominator, 1.0)
    relative = relative_starts[None, :, :]
    ray_distance = _cross_2d(relative, segment) / safe_denominator
    segment_fraction = _cross_2d(relative, ray) / safe_denominator
    valid = (non_parallel & (ray_distance >= 0.0)
             & (ray_distance <= range_max)
             & (segment_fraction >= 0.0) & (segment_fraction <= 1.0))
    candidates = np.where(valid, ray_distance, np.inf)
    scan[:] = np.minimum(scan, np.min(candidates, axis=1))


def _geometry_segments(geometry):
    coordinates = (np.asarray(geometry.exterior.coords, dtype=float)
                   if isinstance(geometry, Polygon)
                   else np.asarray(geometry.coords, dtype=float))
    return coordinates[:-1], coordinates[1:]


def calculate_laser_scan_optimized(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    obstacles: List[Obstacle],
    range_max: float,
    static_obstacles: Iterable,
    angle_min: float = np.deg2rad(-90),
    angle_max: float = np.deg2rad(270),
    angle_increment: float = np.deg2rad(1),
) -> np.ndarray:
    """Compute a scan without constructing Shapely objects per laser beam."""
    relative_angles = np.arange(angle_min, angle_max, angle_increment)
    world_angles = robot_yaw + relative_angles
    directions = np.column_stack((np.cos(world_angles), np.sin(world_angles)))
    origin = np.array([robot_x, robot_y], dtype=float)
    scan = np.full(relative_angles.shape, range_max, dtype=float)

    for obstacle in obstacles:
        if obstacle.shape == ObstacleShape.circle:
            relative_center = np.array(
                [obstacle.x - robot_x, obstacle.y - robot_y], dtype=float
            )
            projection = directions @ relative_center
            perpendicular_squared = (
                relative_center @ relative_center - projection ** 2
            )
            half_chord_squared = obstacle.radius ** 2 - perpendicular_squared
            valid = half_chord_squared >= 0.0
            half_chord = np.sqrt(np.maximum(half_chord_squared, 0.0))
            entry = projection - half_chord
            # Preserve a meaningful zero reading if the sensor is inside a body.
            inside = relative_center @ relative_center <= obstacle.radius ** 2
            distance = np.where(inside, 0.0, entry)
            valid &= (distance >= 0.0) & (distance <= range_max)
            scan = np.minimum(scan, np.where(valid, distance, range_max))
        elif obstacle.shape == ObstacleShape.polygon:
            vertices = transform_vertices_to_world(
                obstacle.vertices, obstacle.x, obstacle.y, obstacle.yaw
            )
            if len(vertices) >= 2:
                closed = np.vstack((vertices, vertices[0]))
                _update_from_segments(
                    scan, directions, origin, range_max,
                    closed[:-1], closed[1:],
                )

    for geometry in static_obstacles:
        if isinstance(geometry, (Polygon, LineString)):
            starts, ends = _geometry_segments(geometry)
            _update_from_segments(
                scan, directions, origin, range_max, starts, ends
            )
    return scan


def compute_rectangle_robot_scan(yaw: float, length: float = 6.0,
                                 width: float = 3.0,
                                 resolution: int = 1) -> List[float]:
    """Compute distances from the robot center to a rectangular footprint."""
    angles = yaw + np.deg2rad(np.arange(0, 360, resolution) + 90)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    # Distance from the center to the first intersection with an axis-aligned
    # rectangle in its own frame. ``yaw`` cancels because both rotate together.
    local_angles = np.deg2rad(np.arange(0, 360, resolution) + 90)
    cosine = np.abs(np.cos(local_angles))
    sine = np.abs(np.sin(local_angles))
    distance_x = np.divide(length / 2, cosine,
                           out=np.full_like(cosine, np.inf), where=cosine > 1e-12)
    distance_y = np.divide(width / 2, sine,
                           out=np.full_like(sine, np.inf), where=sine > 1e-12)
    del directions  # retained above to document the scan convention
    return np.minimum(distance_x, distance_y).tolist()
