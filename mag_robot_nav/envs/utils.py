"""Higher-level utility functions for obstacle generation and transforms."""

import numpy as np
import random
import yaml
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, LineString
from typing import List, Optional, Tuple

from .config import (
    ROI_LENGTH_X, ROI_LENGTH_Y,
    MIN_OBSTACLE_RADIUS, MAX_OBSTACLE_RADIUS,
    EnvType, ObstacleShape,
)
from .geometry import pi_2_pi
from .models import Obstacle


def load_config(config_file: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def generate_convex_polygon(center_x: float, center_y: float,
                            num_points: int,
                            min_radius: float,
                            max_radius: float) -> Tuple[List, float]:
    """Generate a random convex polygon with centroid at (center_x, center_y).

    Returns:
        Tuple of (vertex_list, max_radius).
    """
    points = []
    max_r = min_radius
    for _ in range(num_points):
        rand_r = random.uniform(min_radius, max_radius)
        rand_theta = random.uniform(-np.pi, np.pi)
        points.append((center_x + rand_r * np.cos(rand_theta),
                       center_y + rand_r * np.sin(rand_theta)))
        max_r = max(max_r, rand_r)

    hull = ConvexHull(points)
    vertices = np.array([points[vertex] for vertex in hull.vertices])

    current_center_x = float(np.mean(vertices[:, 0]))
    current_center_y = float(np.mean(vertices[:, 1]))

    # Translate polygon centroid to (center_x, center_y)
    vertices[:, 0] += (center_x - current_center_x)
    vertices[:, 1] += (center_y - current_center_y)

    max_r = float(np.max(np.hypot(
        vertices[:, 0] - current_center_x,
        vertices[:, 1] - current_center_y)))

    return vertices.tolist(), max_r


def transform_vertices_to_world(vertices: List,
                                center_x: float, center_y: float,
                                yaw: float) -> np.ndarray:
    """Transform polygon vertices from local to world coordinate system.

    Returns:
        ndarray of shape (N, 2) in world coordinates.
    """
    vertices_arr = np.array(vertices)
    vertices_homogeneous = np.hstack(
        [vertices_arr, np.ones((vertices_arr.shape[0], 1))])

    translation_matrix = np.array([
        [1, 0, -np.mean(vertices_arr[:, 0])],
        [0, 1, -np.mean(vertices_arr[:, 1])],
        [0, 0, 1]
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    inverse_translation_matrix = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])

    transform_matrix = inverse_translation_matrix @ rotation_matrix @ translation_matrix
    vertices_world_homogeneous = vertices_homogeneous @ transform_matrix.T

    return vertices_world_homogeneous[:, :2]


def generate_random_obstacles(
    num_obstacles: int,
    center: Tuple[float, float],
    radius: float,
    min_radius: float = MIN_OBSTACLE_RADIUS,
    max_radius: float = MAX_OBSTACLE_RADIUS,
    min_vertices: int = 5,
    max_vertices: int = 10,
    obstacle_rng: Optional[np.random.Generator] = None,
    env_type: EnvType = EnvType.circle,
    width: float = ROI_LENGTH_X,
    length: float = ROI_LENGTH_Y,
    static_obstacles: Optional[List] = None,
    obstacle_shape: ObstacleShape = ObstacleShape.circle,
) -> Optional[List[Obstacle]]:
    """Randomly generate obstacles with support for circle and polygon shapes.

    Returns None if obstacles cannot be placed safely.
    """
    if num_obstacles < 0:
        raise ValueError("num_obstacles must be non-negative")
    if num_obstacles == 0:
        return []

    zoom = 1.5  # polygon radius zoom factor
    circle_mindist_to_center_ratio = 0.0
    circle_maxdist_to_center_ratio = 0.95
    square_maxdist_to_center_ratio = 0.95
    large_env_mindist_to_center_ratio = 0.05
    large_env_maxdist_to_center_ratio = 0.95

    obstacles = []
    if obstacle_rng is not None:
        yaws = obstacle_rng.uniform(-np.pi, np.pi, num_obstacles)

        if env_type == EnvType.circle:
            ratios = obstacle_rng.uniform(0, 1, num_obstacles)
            thetas = (np.arange(0, 2 * np.pi, 2 * np.pi / num_obstacles)
                      + ratios * 2 * np.pi / num_obstacles)
            xs = (center[0]
                  + obstacle_rng.uniform(radius * circle_mindist_to_center_ratio,
                                         radius * circle_maxdist_to_center_ratio,
                                         num_obstacles)
                  * np.cos(thetas))
            ys = (center[1]
                  + obstacle_rng.uniform(radius * circle_mindist_to_center_ratio,
                                         radius * circle_maxdist_to_center_ratio,
                                         num_obstacles)
                  * np.sin(thetas))
        elif env_type == EnvType.square:
            xs = obstacle_rng.uniform(
                center[0] - radius * square_maxdist_to_center_ratio,
                center[0] + radius * square_maxdist_to_center_ratio,
                num_obstacles)
            ys = obstacle_rng.uniform(
                center[1] - radius * square_maxdist_to_center_ratio,
                center[1] + radius * square_maxdist_to_center_ratio,
                num_obstacles)
        elif env_type == EnvType.vessel:
            xs = obstacle_rng.uniform(
                width * large_env_mindist_to_center_ratio,
                width * large_env_maxdist_to_center_ratio, num_obstacles)
            ys = obstacle_rng.uniform(
                length * large_env_mindist_to_center_ratio,
                length * large_env_maxdist_to_center_ratio, num_obstacles)
        elif env_type == EnvType.maze:
            xs = obstacle_rng.uniform(
                width * large_env_mindist_to_center_ratio,
                width * large_env_maxdist_to_center_ratio, num_obstacles)
            ys = obstacle_rng.uniform(
                length * large_env_mindist_to_center_ratio,
                length * large_env_maxdist_to_center_ratio, num_obstacles)

        radii = obstacle_rng.uniform(min_radius, max_radius, num_obstacles)

        for i in range(num_obstacles):
            if obstacle_shape == ObstacleShape.circle:
                obstacle = Obstacle(id=i, shape=obstacle_shape,
                                    radius=radii[i],
                                    x=xs[i], y=ys[i], yaw=yaws[i])
            elif obstacle_shape == ObstacleShape.polygon:
                random_vertices, max_r = generate_convex_polygon(
                    xs[i], ys[i],
                    int(obstacle_rng.uniform(min_vertices, max_vertices)),
                    min_radius * zoom, max_radius * zoom)
                obstacle = Obstacle(id=i, shape=obstacle_shape,
                                    vertices=random_vertices, radius=max_r,
                                    x=xs[i], y=ys[i], yaw=yaws[i])
            obstacles.append(obstacle)

        # Validate: obstacle must be inside tank and not too close to static obstacles
        for obs in obstacles:
            if not static_obstacles[0].contains(Point(obs.x, obs.y)):
                return None
            for static_obs in static_obstacles:
                if isinstance(static_obs, Polygon):
                    dist_to_obstacle = Point(obs.x, obs.y).distance(
                        static_obs.exterior)
                else:
                    dist_to_obstacle = static_obs.distance(Point(obs.x, obs.y))
                if dist_to_obstacle < obs.radius:
                    return None

    else:
        # Totally random through 'random' module, not using seed
        obstacles_count = 0
        while obstacles_count < num_obstacles:
            yaw = random.uniform(-np.pi, np.pi)

            if env_type == EnvType.circle:
                theta = random.uniform(
                    obstacles_count * np.pi * 2 / num_obstacles,
                    (obstacles_count + 1) * np.pi * 2 / num_obstacles)
                x = (center[0]
                     + random.uniform(radius * circle_mindist_to_center_ratio,
                                      radius * circle_maxdist_to_center_ratio)
                     * np.cos(theta))
                y = (center[1]
                     + random.uniform(radius * circle_mindist_to_center_ratio,
                                      radius * circle_maxdist_to_center_ratio)
                     * np.sin(theta))
            elif env_type == EnvType.square:
                x = random.uniform(
                    center[0] - radius * square_maxdist_to_center_ratio,
                    center[0] + radius * square_maxdist_to_center_ratio)
                y = random.uniform(
                    center[1] - radius * square_maxdist_to_center_ratio,
                    center[1] + radius * square_maxdist_to_center_ratio)
            elif env_type == EnvType.vessel:
                x = random.uniform(width * large_env_mindist_to_center_ratio,
                                   width * large_env_maxdist_to_center_ratio)
                y = random.uniform(length * large_env_mindist_to_center_ratio,
                                   length * large_env_maxdist_to_center_ratio)
            elif env_type == EnvType.maze:
                x = random.uniform(width * large_env_mindist_to_center_ratio,
                                   width * large_env_maxdist_to_center_ratio)
                y = random.uniform(length * large_env_mindist_to_center_ratio,
                                   length * large_env_maxdist_to_center_ratio)

            rad = random.uniform(min_radius, max_radius)

            if not static_obstacles[0].contains(Point(x, y)):
                continue

            safe = True
            for static_obs in static_obstacles:
                if isinstance(static_obs, Polygon):
                    dist_to_obstacle = Point(x, y).distance(
                        static_obs.exterior)
                else:
                    dist_to_obstacle = static_obs.distance(Point(x, y))
                if dist_to_obstacle < rad:
                    safe = False
                    break
            if not safe:
                continue

            if obstacle_shape == ObstacleShape.circle:
                obstacle = Obstacle(id=obstacles_count, shape=obstacle_shape,
                                    radius=rad, x=x, y=y, yaw=yaw)
            elif obstacle_shape == ObstacleShape.polygon:
                random_vertices, max_r = generate_convex_polygon(
                    x, y, int(random.uniform(min_vertices, max_vertices)),
                    min_radius * zoom, max_radius * zoom)
                obstacle = Obstacle(id=obstacles_count, shape=obstacle_shape,
                                    vertices=random_vertices, radius=max_r,
                                    x=x, y=y, yaw=yaw)
            obstacles.append(obstacle)
            obstacles_count += 1

    return obstacles
