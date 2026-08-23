import numpy as np
import pytest
from shapely.geometry import LineString

from mag_robot_nav.envs.config import ObstacleShape
from mag_robot_nav.envs.laser import calculate_laser_scan_optimized
from mag_robot_nav.envs.models import Obstacle


def one_ray(obstacles=(), static_obstacles=()):
    return calculate_laser_scan_optimized(
        0.0, 0.0, 0.0, list(obstacles), 10.0, list(static_obstacles),
        angle_min=0.0, angle_max=np.deg2rad(1),
        angle_increment=np.deg2rad(1),
    )[0]


def test_circle_intersection_is_analytic():
    obstacle = Obstacle(
        id=1, shape=ObstacleShape.circle, x=5.0, y=0.0, radius=1.0
    )
    assert one_ray([obstacle]) == pytest.approx(4.0)


def test_static_line_intersection_is_analytic():
    assert one_ray(static_obstacles=[LineString([(3, -1), (3, 1)])]) == pytest.approx(3.0)


def test_objects_outside_range_leave_maximum_reading():
    obstacle = Obstacle(
        id=1, shape=ObstacleShape.circle, x=20.0, y=0.0, radius=1.0
    )
    assert one_ray([obstacle]) == pytest.approx(10.0)
