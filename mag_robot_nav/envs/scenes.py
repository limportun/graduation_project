"""Static scene construction independent of the Gym environment lifecycle."""

from dataclasses import dataclass
from typing import Callable, List

from shapely.geometry import LineString, Point, Polygon

from .config import EnvType, ROI_LENGTH_X, ROI_LENGTH_Y
from .models import Obstacle


@dataclass(frozen=True)
class StaticScene:
    width: float
    length: float
    obstacles: List


def build_static_scene(env_type: EnvType) -> StaticScene:
    """Create the immutable geometry for one environment layout."""
    width, length = ROI_LENGTH_X, ROI_LENGTH_Y
    if env_type == EnvType.circle:
        obstacles = [Point(width / 2, length / 2).buffer(length / 2)]
    elif env_type == EnvType.square:
        cx, cy, radius = width / 2, length / 2, length / 2
        obstacles = [
            Polygon([(cx - radius, 0), (cx + radius, 0),
                     (cx + radius, length), (cx - radius, length)]),
            LineString([(cx - length / 6, cy + radius / 3),
                        (cx - length / 6, length)]),
            LineString([(cx + length / 6, 0),
                        (cx + length / 6, radius / 3)]),
            LineString([(0, length / 3), (cx - length / 6, length / 3)]),
            LineString([(cx + length / 6, length * 2 / 3),
                        (width, length * 2 / 3)]),
        ]
    elif env_type == EnvType.vessel:
        width, length = width * 2, length * 2
        obstacles = [Polygon([
            (width / 20, length * 2 / 5),
            (width / 3, length * 2 / 5),
            (width * 2 / 3, length * 2 / 3),
            (width * 5 / 6, length * 19 / 20),
            (width * 7 / 8, length * 17 / 20),
            (width * 9 / 12, length * 5 / 8),
            (width * 37 / 40, length * 15 / 20),
            (width * 19 / 20, length * 2 / 3),
            (width * 2 / 5, length * 5 / 18),
            (width * 2 / 3, length * 5 / 32),
            (width * 2 / 3, length / 20),
            (width / 3, length / 5),
            (width / 20, length / 5),
        ])]
    elif env_type == EnvType.maze:
        width = length = ROI_LENGTH_X * 1.5
        w6, l6 = width / 6, length / 6
        segments = [
            ((0, 5), (1, 5)), ((2, 5), (3, 5)), ((3, 3), (3, 6)),
            ((0, 4), (1, 4)), ((1, 3), (1, 4)), ((2, 3), (2, 4)),
            ((2, 3), (3, 3)), ((1, 1), (1, 2)), ((2, 1), (3, 1)),
            ((1, 2), (5, 2)), ((2, 0), (2, 1)), ((4, 0), (4, 1)),
            ((5, 1), (5, 2)), ((4, 4), (5, 4)), ((5, 3), (6, 3)),
            ((5, 5), (6, 5)), ((4, 3), (4, 5)),
        ]
        obstacles = [Polygon([(0, 0), (width, 0), (width, length), (0, length)])]
        obstacles.extend(LineString([
            (start[0] * w6, start[1] * l6),
            (end[0] * w6, end[1] * l6),
        ]) for start, end in segments)
    else:
        obstacles = []
    return StaticScene(width=width, length=length, obstacles=obstacles)


def discretize_static_obstacles(obstacles, spacing: float):
    """Return collision-checking circles and their coordinate arrays."""
    circles, x_coordinates, y_coordinates = [], [], []
    for obstacle in obstacles:
        if isinstance(obstacle, Polygon):
            coordinates = list(obstacle.exterior.coords)
            edges = [LineString(coordinates[index:index + 2])
                     for index in range(len(coordinates) - 1)]
        elif isinstance(obstacle, LineString):
            edges = [obstacle]
        else:
            continue
        for edge in edges:
            for index in range(int(edge.length // spacing) + 1):
                point = edge.interpolate(index * spacing)
                x_coordinates.append(point.x)
                y_coordinates.append(point.y)
                circles.append(Obstacle(
                    id=-1, shape="circle", x=point.x, y=point.y,
                    radius=spacing,
                ))
    return circles, x_coordinates, y_coordinates


def build_pixel_obstacles(obstacles, world_to_pixel: Callable[[float], int]):
    """Convert immutable scene geometry once for rendering."""
    converted = []
    for obstacle in obstacles:
        if isinstance(obstacle, Polygon):
            converted.append(Polygon([
                (world_to_pixel(x), world_to_pixel(y))
                for x, y in obstacle.exterior.coords
            ]))
        elif isinstance(obstacle, LineString):
            converted.append(LineString([
                (world_to_pixel(x), world_to_pixel(y))
                for x, y in obstacle.coords
            ]))
    return converted
