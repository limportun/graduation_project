"""Low-level geometry/math utilities with no project-internal dependencies."""

import numpy as np


def pi_2_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def euclidean_distance(x1: float, y1: float,
                       x2: float, y2: float) -> float:
    """Compute the Euclidean distance between two points."""
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def point_to_line_distance(px: float, py: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> float:
    """Compute the distance from a point to a line segment."""
    line_len = euclidean_distance(x1, y1, x2, y2)
    if line_len == 0:
        return euclidean_distance(px, py, x1, y1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1))
                   / line_len ** 2))
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    return euclidean_distance(px, py, projection_x, projection_y)


def transform_to_robot_frame(robot_pose: np.ndarray,
                             target_poses: np.ndarray) -> np.ndarray:
    """Transform target poses from global frame to robot's coordinate frame.

    Parameters:
        robot_pose: shape (3,), containing [x, y, yaw] in the global frame.
        target_poses: shape (3, n), containing the poses of target objects.

    Returns:
        relative_poses: shape (3, n), relative poses in the robot's frame.
    """
    x_r, y_r, yaw_r = robot_pose

    cos_r, sin_r = np.cos(yaw_r), np.sin(yaw_r)
    R_inv = np.array([[cos_r, sin_r], [-sin_r, cos_r]])

    target_positions = target_poses[:2, :]
    target_yaws = target_poses[2, :]

    translation = target_positions - np.array([[x_r], [y_r]])
    relative_positions = R_inv @ translation
    relative_yaws = target_yaws - yaw_r

    return np.concatenate(
        (relative_positions, relative_yaws[np.newaxis, :]), axis=0)
