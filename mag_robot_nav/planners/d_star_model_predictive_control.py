"""D* global planning with model-predictive path tracking.

This module preserves the rule-based baseline used by the authors' original
evaluation script while removing its image-sized intermediate buffers, global
state, plotting, and environment-specific side effects.
"""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from mag_robot_nav.envs.config import ROI_LENGTH_X

from .d_star import Dstar, Map
from .model_predictive_control import ModelPredictiveControl
from .probabilistic_road_map import prm_planning


class DStarModelPredictiveControl:
    """Replan with D* every step and track the path with iterative MPC."""

    def __init__(self, base_grid_width: int = 80, lookahead_steps: int = 4,
                 verbose: bool = False):
        if base_grid_width <= 1:
            raise ValueError("base_grid_width must be greater than one")
        if lookahead_steps < 0:
            raise ValueError("lookahead_steps must be non-negative")
        self.base_grid_width = int(base_grid_width)
        self.lookahead_steps = int(lookahead_steps)
        self.verbose = bool(verbose)
        self.tracker = None
        self._static_grid_key = None
        self._static_occupancy = None
        self._scale = None
        self.last_used_fallback = False
        self.last_timings = {}

    def reset(self) -> None:
        self.tracker = None
        self.last_used_fallback = False
        self.last_timings = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _prepare_static_occupancy(self, env) -> None:
        resolution_x = max(
            2, int(self.base_grid_width / ROI_LENGTH_X * env.env_width)
        )
        resolution_y = max(
            2, int(resolution_x / env.env_width * env.env_length)
        )
        key = (env.env_type, env.env_width, env.env_length,
               resolution_x, resolution_y)
        if key == self._static_grid_key:
            return

        scale = resolution_x / env.env_width
        clearance = env.safe_dist * 9 + env.robot_width / 2
        thickness = max(1, int(round(2 * clearance * scale)))
        occupancy = np.zeros((resolution_y, resolution_x), dtype=np.uint8)

        for obstacle in env.static_obstacles:
            if isinstance(obstacle, Polygon) and len(obstacle.exterior.coords) >= 32:
                center = (
                    int(round(obstacle.centroid.x * scale)),
                    int(round(obstacle.centroid.y * scale)),
                )
                radius = int(round(np.sqrt(obstacle.area / np.pi) * scale))
                cv2.circle(occupancy, center, radius, 1, thickness=thickness)
            elif isinstance(obstacle, Polygon):
                points = np.rint(
                    np.asarray(obstacle.exterior.coords) * scale
                ).astype(np.int32)
                cv2.polylines(
                    occupancy, [points], isClosed=True, color=1,
                    thickness=thickness + 1,
                )
            elif isinstance(obstacle, LineString):
                coordinates = np.asarray(obstacle.coords)
                start = tuple(np.rint(coordinates[0] * scale).astype(int))
                end = tuple(np.rint(coordinates[-1] * scale).astype(int))
                cv2.line(occupancy, start, end, 1, thickness=thickness)

        self._static_grid_key = key
        self._static_occupancy = occupancy
        self._scale = scale

    def _dynamic_occupancy(self, env) -> np.ndarray:
        self._prepare_static_occupancy(env)
        occupancy = self._static_occupancy.copy()
        clearance = env.safe_dist * 9 + env.robot_width / 2
        for obstacle in env.obstacles:
            center = (
                int(round(obstacle.x * self._scale)),
                int(round(obstacle.y * self._scale)),
            )
            radius = max(
                1, int(round((obstacle.radius + clearance) * self._scale))
            )
            cv2.circle(occupancy, center, radius, 1, thickness=-1)
        return occupancy

    def _world_to_map(self, value: float, limit: int) -> int:
        return int(np.clip(int(value * self._scale), 0, limit - 1))

    def _map_to_world(self, value: int) -> float:
        return value / self._scale

    def _prm_fallback(self, env):
        obstacle_x = env.static_obstacle_x + [item.x for item in env.obstacles]
        obstacle_y = env.static_obstacle_y + [item.y for item in env.obstacles]
        random_seed = int(
            env._seed * env.capacity_level_2 * env.capacity_level_3
            + env.episode_count * env.capacity_level_3
            + env.timestep
        )
        path_x, path_y = prm_planning(
            env.robot.x, env.robot.y, env.goal.x, env.goal.y,
            obstacle_x, obstacle_y, env.robot_radius / 2,
            rng=np.random.default_rng(random_seed),
            show_animation=False, verbose=self.verbose,
        )
        if not path_x or not path_y:
            self._log("D* and PRM failed; falling back to the direct path")
            return [env.robot.x, env.goal.x], [env.robot.y, env.goal.y]

        # PythonRobotics PRM reconstructs from goal to start.
        return list(reversed(path_x)), list(reversed(path_y))

    def plan_path(self, env):
        """Build the inflated occupancy grid and return a world-frame path."""
        started_at = perf_counter()
        occupancy = self._dynamic_occupancy(env)
        occupancy_ready_at = perf_counter()
        resolution_y, resolution_x = occupancy.shape
        start = (
            self._world_to_map(env.robot.x, resolution_x),
            self._world_to_map(env.robot.y, resolution_y),
        )
        goal = (
            self._world_to_map(env.goal.x, resolution_x),
            self._world_to_map(env.goal.y, resolution_y),
        )

        grid_map = Map(resolution_x, resolution_y)
        obstacle_y, obstacle_x = np.nonzero(occupancy)
        grid_map.set_obstacle(list(zip(obstacle_x.tolist(),
                                       obstacle_y.tolist())))
        path_x, path_y = Dstar(grid_map).run(
            grid_map.map[start[0]][start[1]],
            grid_map.map[goal[0]][goal[1]],
        )
        search_finished_at = perf_counter()

        self.last_used_fallback = not path_x or not path_y
        if self.last_used_fallback:
            path_x, path_y = self._prm_fallback(env)
        else:
            path_x = [self._map_to_world(index) for index in path_x]
            path_y = [self._map_to_world(index) for index in path_y]
        finished_at = perf_counter()
        self.last_timings = {
            "occupancy": occupancy_ready_at - started_at,
            "dstar": search_finished_at - occupancy_ready_at,
            "fallback": finished_at - search_finished_at,
        }
        return path_x, path_y

    @staticmethod
    def _prepare_tracker_path(path_x, path_y, goal_x, goal_y):
        points = []
        for x, y in zip(path_x, path_y):
            point = (float(x), float(y))
            if not points or not np.allclose(point, points[-1], atol=1e-9):
                points.append(point)
        goal = (float(goal_x), float(goal_y))
        if not points or not np.allclose(goal, points[-1], atol=1e-9):
            points.append(goal)
        if len(points) == 1:
            points.append((points[0][0] + 1e-3, points[0][1]))
        return [point[0] for point in points], [point[1] for point in points]

    def predict(self, env) -> np.ndarray:
        """Return the legacy normalized action used by the original baseline."""
        started_at = perf_counter()
        path_x, path_y = self.plan_path(env)
        path_x, path_y = self._prepare_tracker_path(
            path_x, path_y, env.goal.x, env.goal.y
        )
        env.planning_path = list(zip(path_x, path_y))
        env.pixel_planning_path = [
            (env.world2pixel(x), env.world2pixel(y))
            for x, y in env.planning_path
        ]

        lookahead = min(len(path_x) - 1, self.lookahead_steps)
        env.sub_goal = env.goal.copy()
        env.sub_goal.x, env.sub_goal.y = path_x[lookahead], path_y[lookahead]

        if self.tracker is None:
            self.tracker = ModelPredictiveControl(
                env.robot.x, env.robot.y, env.robot.yaw, env.robot.vx,
                path_x, path_y,
            )
        else:
            self.tracker.update(
                env.robot.x, env.robot.y, env.robot.vx, env.robot.yaw,
                path_x, path_y, gx=env.goal.x, gy=env.goal.y,
            )

        velocity, angular_velocity = self.tracker.plan(
            env.robot.x, env.robot.y, env.robot.vx, env.robot.yaw
        )
        self.last_timings["mpc"] = perf_counter() - started_at - sum(
            self.last_timings.values()
        )
        self.last_timings["total"] = perf_counter() - started_at
        if self.tracker.reference is not None:
            env.sub_goal.x = float(self.tracker.reference[0, -1])
            env.sub_goal.y = float(self.tracker.reference[1, -1])

        # The published baseline fed MPC's physical outputs directly to the
        # environment action input. Preserve that behavior for comparability.
        return np.clip(
            np.asarray([velocity, angular_velocity], dtype=np.float32),
            -1.0, 1.0,
        )


__all__ = ["DStarModelPredictiveControl"]
