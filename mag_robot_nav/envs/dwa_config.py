"""Configuration classes for Dynamic Window Approach (DWA) planner."""

import numpy as np
from .config import (
    DT, SAFE_DIST, ARRIVAL_THRESHOLD,
    ROBOT_WIDTH, ROBOT_LENGTH,
    MIN_VELOCITY, MAX_VELOCITY,
    MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    MAX_ACCELERATION, MAX_ANGULAR_ACCELERATION,
)
from ..planners.dynamic_window_approach import RobotType


class Config:
    """DWA simulation parameters for the robot."""

    def __init__(self, obstacles: np.ndarray):
        self.max_speed: float = MAX_VELOCITY
        self.min_speed: float = MIN_VELOCITY
        self.max_yaw_rate: float = MAX_ANGULAR_VELOCITY
        self.max_accel: float = MAX_ACCELERATION
        self.max_delta_yaw_rate: float = MAX_ANGULAR_ACCELERATION
        self.v_resolution: float = 0.075 * (MAX_VELOCITY - MIN_VELOCITY)
        self.yaw_rate_resolution: float = 0.01 * (
            MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)
        self.dt: float = DT
        self.predict_time: float = self.dt * 20
        self.to_goal_cost_gain: float = 0.03
        self.speed_cost_gain: float = 0.00
        self.obstacle_cost_gain: float = 1.0
        self.robot_stuck_flag_const: float = 0.001
        self.safe_dist: float = SAFE_DIST

        self._robot_type: RobotType = RobotType.rectangle
        self.robot_width: float = ROBOT_WIDTH
        self.robot_length: float = ROBOT_LENGTH
        self.ob: np.ndarray = obstacles

    @property
    def robot_type(self) -> RobotType:
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value: RobotType) -> None:
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


class GoalConfig:
    """DWA simulation parameters for the goal (scaled-down robot)."""

    def __init__(self, obstacles: np.ndarray, ratio: float = 0.5):
        self.max_speed: float = MAX_VELOCITY * ratio
        self.min_speed: float = MIN_VELOCITY * ratio
        self.max_yaw_rate: float = MAX_ANGULAR_VELOCITY * ratio
        self.max_accel: float = MAX_ACCELERATION * ratio
        self.max_delta_yaw_rate: float = MAX_ANGULAR_ACCELERATION * ratio
        self.v_resolution: float = 0.05 * (MAX_VELOCITY - MIN_VELOCITY)
        self.yaw_rate_resolution: float = 0.05 * (
            MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)
        self.dt: float = DT
        self.predict_time: float = self.dt * 20
        self.to_goal_cost_gain: float = 0.0
        self.speed_cost_gain: float = 0.3
        self.obstacle_cost_gain: float = 1.0
        self.robot_stuck_flag_const: float = 0.001
        self.safe_dist: float = SAFE_DIST

        self._robot_type: RobotType = RobotType.circle
        self.robot_radius: float = ARRIVAL_THRESHOLD
        self.ob: np.ndarray = obstacles

    @property
    def robot_type(self) -> RobotType:
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value: RobotType) -> None:
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value
