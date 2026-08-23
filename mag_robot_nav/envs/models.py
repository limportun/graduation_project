"""Data model classes for the MagRobot navigation environment."""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import (
    MAX_ACCELERATION, MAX_ANGULAR_ACCELERATION,
    MIN_VELOCITY, MAX_VELOCITY,
    MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    MIN_OBSTACLE_ANGULAR_VELOCITY, MAX_OBSTACLE_ANGULAR_VELOCITY,
    MAX_OBSTACLE_VELOCITY,
    MAX_OBSTACLE_ACCELERATION, MAX_OBSTACLE_ANGULAR_ACCELERATION,
    ObstacleShape,
)
from .geometry import pi_2_pi, euclidean_distance, point_to_line_distance


@dataclass
class Command:
    """Dataclass for storing motion commands."""
    target_vx: float
    target_vy: float
    target_vw: float
    timestamp: float


class Robot:
    """Mobile robot with velocity-constrained kinematics and command delay."""

    def __init__(self, x: float, y: float, yaw: float,
                 vx: float, vy: float, vw: float,
                 delay: float,
                 max_accel: float = MAX_ACCELERATION,
                 max_angular_accel: float = MAX_ANGULAR_ACCELERATION):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.vw = vw
        self.delay = delay
        self.max_accel = max_accel
        self.max_angular_accel = max_angular_accel

        self.command_queue: deque = deque()
        self.current_time: float = 0.0

    def queue_command(self, target_vx: float, target_vy: float,
                      target_vw: float) -> None:
        """Enqueue a new command."""
        command = Command(target_vx, target_vy, target_vw, self.current_time)
        self.command_queue.append(command)

    def get_delayed_command(self) -> Tuple[float, float, float]:
        """Get the delayed command to execute."""
        target_time = self.current_time - self.delay

        while self.command_queue and self.command_queue[0].timestamp < target_time:
            self.command_queue.popleft()

        if self.current_time < self.delay:
            return (0.0, 0.0, 0.0)

        return (self.command_queue[0].target_vx,
                self.command_queue[0].target_vy,
                self.command_queue[0].target_vw)

    def update(self, target_vx: float, target_vy: float,
               target_vw: float, dt: float) -> None:
        """Step the robot forward with acceleration-limited kinematics."""
        self.current_time += dt
        self.queue_command(target_vx, target_vy, target_vw)
        delayed_vx, delayed_vy, delayed_vw = self.get_delayed_command()

        ax = delayed_vx - self.vx
        ay = delayed_vy - self.vy
        aw = delayed_vw - self.vw

        # Constrain linear acceleration
        accel = np.sqrt(ax ** 2 + ay ** 2)
        if accel > self.max_accel:
            ax = (ax / accel) * self.max_accel
            ay = (ay / accel) * self.max_accel

        # Constrain angular acceleration
        if np.abs(aw) > self.max_angular_accel:
            aw = np.sign(aw) * self.max_angular_accel

        # Step forward with constrained acceleration
        self.vx += ax * dt
        self.vy += ay * dt
        self.vw += aw * dt
        # Constrain velocities
        self.vx = np.clip(self.vx, MIN_VELOCITY, MAX_VELOCITY)
        self.vy = np.clip(self.vy, MIN_VELOCITY, MAX_VELOCITY)
        self.vw = np.clip(self.vw, MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # Step forward with constrained velocity
        self.x = self.x + self.vx * dt * np.cos(self.yaw) - self.vy * dt * np.sin(self.yaw)
        self.y = self.y + self.vx * dt * np.sin(self.yaw) + self.vy * dt * np.cos(self.yaw)
        self.yaw = pi_2_pi(self.yaw + self.vw * dt)

    def copy(self) -> 'Robot':
        """Create a copy of the robot with the same state and command queue."""
        copied_robot = Robot(
            x=self.x, y=self.y, yaw=self.yaw,
            vx=self.vx, vy=self.vy, vw=self.vw,
            delay=self.delay,
            max_accel=self.max_accel,
            max_angular_accel=self.max_angular_accel)
        copied_robot.command_queue = self.command_queue.copy()
        copied_robot.current_time = self.current_time
        return copied_robot


class Obstacle:
    """Dynamic obstacle with circle or polygon shape."""

    def __init__(self, id: int, shape: ObstacleShape,
                 vertices: Optional[List] = None,
                 radius: Optional[float] = None,
                 x: Optional[float] = None, y: Optional[float] = None,
                 yaw: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        self.id = id
        self.shape = shape
        self.vertices = vertices if vertices else []
        self.radius = radius
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy

        self.vw: float = 0.0
        self.max_accel: float = MAX_OBSTACLE_ACCELERATION
        self.max_angular_accel: float = MAX_OBSTACLE_ANGULAR_ACCELERATION

    def update(self, target_vx: float, target_vy: float,
               target_vw: float, dt: float) -> None:
        """Step the obstacle forward with acceleration-limited kinematics."""
        ax = target_vx - self.vx
        ay = target_vy - self.vy
        aw = target_vw - self.vw

        # Constrain linear acceleration
        accel = np.sqrt(ax ** 2 + ay ** 2)
        if accel > self.max_accel:
            ax = (ax / accel) * self.max_accel
            ay = (ay / accel) * self.max_accel

        # Constrain angular acceleration
        if np.abs(aw) > self.max_angular_accel:
            aw = np.sign(aw) * self.max_angular_accel

        # Update constrained velocity
        self.vx += ax * dt
        self.vy += ay * dt
        self.vw += aw * dt

        # Constrain velocities
        self.vx = np.clip(self.vx, 0, MAX_OBSTACLE_VELOCITY)
        self.vy = np.clip(self.vy, 0, MAX_OBSTACLE_VELOCITY)
        self.vw = np.clip(self.vw, MIN_OBSTACLE_ANGULAR_VELOCITY,
                          MAX_OBSTACLE_ANGULAR_VELOCITY)

        # Update pose
        self.x = self.x + self.vx * dt * np.cos(self.yaw) - self.vy * dt * np.sin(self.yaw)
        self.y = self.y + self.vx * dt * np.sin(self.yaw) + self.vy * dt * np.cos(self.yaw)
        self.yaw = pi_2_pi(self.yaw + self.vw * dt)

    def get_distance_to_robot(self, robot_x: float, robot_y: float) -> float:
        """Compute the shortest distance from robot to obstacle."""
        if self.shape == ObstacleShape.circle:
            center_dist = euclidean_distance(self.x, self.y, robot_x, robot_y)
            return max(0, center_dist - self.radius)
        elif self.shape == ObstacleShape.polygon:
            min_distance = float('inf')
            for i in range(len(self.vertices)):
                x1, y1 = self.vertices[i]
                x2, y2 = self.vertices[(i + 1) % len(self.vertices)]
                dist = point_to_line_distance(robot_x, robot_y, x1, y1, x2, y2)
                min_distance = min(min_distance, dist)
            return min_distance
        return float('inf')


class Goal(Robot):
    """Goal point that may optionally move."""

    def __init__(self, x: float, y: float, yaw: float = 0.0,
                 vx: float = 0.0, vy: float = 0.0, vw: float = 0.0,
                 delay: float = 0.0, static: bool = True):
        super().__init__(x, y, yaw, vx, vy, vw, delay)
        self.static = static
