"""Rendering utilities for converting obstacle data to RGB images."""

import cv2
import numpy as np
from shapely.geometry import Polygon, LineString
from typing import List, Optional, Tuple



# ---------------------------------------------------------------------------
#  Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_robot_on_canvas(canvas: np.ndarray,
                          center: Tuple[int, int],
                          yaw: float,
                          length: float,
                          width: float,
                          front_color: Tuple[int, int, int],
                          rear_color: Tuple[int, int, int]) -> None:
    """Draw a two-color filled rotated rectangle (front + rear halves)."""
    cx, cy = center
    yaw_deg = np.rad2deg(yaw)

    # Front half
    front_cx = int(cx + length / 4 * np.cos(yaw))
    front_cy = int(cy + length / 4 * np.sin(yaw))
    rect_front = ((front_cx, front_cy), (float(length) / 2, float(width)),
                  yaw_deg)
    box_front = cv2.boxPoints(rect_front).astype(np.int32)
    cv2.fillPoly(canvas, [box_front], color=front_color)

    # Rear half
    rear_cx = int(cx - length / 4 * np.cos(yaw))
    rear_cy = int(cy - length / 4 * np.sin(yaw))
    rect_rear = ((rear_cx, rear_cy), (float(length) / 2, float(width)),
                 yaw_deg)
    box_rear = cv2.boxPoints(rect_rear).astype(np.int32)
    cv2.fillPoly(canvas, [box_rear], color=rear_color)


def _draw_goal_on_canvas(canvas: np.ndarray,
                         center: Tuple[int, int],
                         radius: int,
                         yaw: float,
                         color: Tuple[int, int, int] = (30, 200, 10),
                         thickness: int = 5) -> None:
    """Draw goal as a circle with direction arrow."""
    cv2.circle(canvas, center, radius, color=color, thickness=thickness)
    arrow_len = radius * 0.8
    cv2.arrowedLine(
        canvas, center,
        (int(center[0] + arrow_len * np.cos(yaw)),
         int(center[1] + arrow_len * np.sin(yaw))),
        color=(0, 150, 0), thickness=3)


def _draw_tank_edge(canvas: np.ndarray,
                    static_obstacles: List,
                    color: Tuple[int, int, int] = (150, 0, 0),
                    thickness: int = 8) -> None:
    """Draw static obstacle boundaries."""
    for obs in static_obstacles:
        if isinstance(obs, Polygon) and len(obs.exterior.coords) >= 32:
            cv2.circle(canvas,
                       (int(obs.centroid.x), int(obs.centroid.y)),
                       int(np.sqrt(obs.area / np.pi)),
                       color=color, thickness=thickness)
        elif isinstance(obs, Polygon):
            coords = np.array(obs.exterior.coords, dtype=np.int32)
            cv2.polylines(canvas, [coords], isClosed=True,
                          color=color, thickness=thickness)
        elif isinstance(obs, LineString):
            cv2.line(canvas,
                     (int(obs.coords[0][0]), int(obs.coords[0][1])),
                     (int(obs.coords[1][0]), int(obs.coords[1][1])),
                     color=color, thickness=thickness)


# ---------------------------------------------------------------------------
#  Public rendering functions
# ---------------------------------------------------------------------------

def convert_obstacles_to_image_RGB(
    center_xs: np.ndarray, center_ys: np.ndarray, radii: np.ndarray,
    vertices_list: Optional[List],
    robot_width: float, robot_length: float,
    robot_center: Tuple[int, int], robot_yaw: float,
    goal_center: Tuple[int, int], goal_radius: int, goal_yaw: float,
    static_obstacles: List,
    image_width: int = 1920, image_height: int = 1080,
    pixel_planned_path: List = [(0, 0)],
    pixel_sub_goal: Optional[Tuple[int, int]] = None,
    robot_traj: List = [],
) -> np.ndarray:
    """Render obstacles with planned path and sub-goal for evaluation."""
    map_image = np.full((image_height, image_width, 3), 240, dtype=np.uint8)

    _draw_robot_on_canvas(map_image, robot_center, robot_yaw,
                          robot_length, robot_width,
                          front_color=(0, 30, 200), rear_color=(200, 30, 0))

    _draw_goal_on_canvas(map_image, goal_center, goal_radius, goal_yaw)
    cv2.circle(map_image, goal_center, 8, color=(0, 150, 0), thickness=-1)

    _draw_tank_edge(map_image, static_obstacles)

    # Sub-goal and planned path
    if pixel_sub_goal is not None and len(pixel_planned_path) > 1:
        cv2.circle(map_image, pixel_sub_goal, radius=goal_radius,
                   color=(200, 30, 10), thickness=6)
    if len(pixel_planned_path) > 1:
        for i in range(len(pixel_planned_path) - 2):
            cv2.line(map_image,
                     (int(pixel_planned_path[i][0]), int(pixel_planned_path[i][1])),
                     (int(pixel_planned_path[i + 1][0]),
                      int(pixel_planned_path[i + 1][1])),
                     color=(150, 150, 0), thickness=7)

    if vertices_list is None:
        vertices_list = [None] * len(center_xs)
    for x, y, r, ves in zip(center_xs, center_ys, radii, vertices_list):
        if ves is None:
            cv2.circle(map_image, (x, y), r, (80, 80, 80), -1)
        else:
            cv2.fillPoly(map_image, [ves], (80, 80, 80))

    for i in range(len(robot_traj) - 1):
        cv2.line(map_image,
                 (robot_traj[i][0], robot_traj[i][1]),
                 (robot_traj[i + 1][0], robot_traj[i + 1][1]),
                 (50, 50, 100), thickness=3)

    return map_image
