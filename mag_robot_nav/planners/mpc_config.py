"""Typed configuration for the iterative linear MPC planner."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelPredictiveControlConfig:
    horizon: int = 2
    time_step: float = 0.2
    max_iterations: int = 5
    convergence_threshold: float = 0.1
    max_angular_velocity: float = np.deg2rad(45.0)
    max_angular_acceleration: float = np.deg2rad(90.0)
    max_speed: float = 2.0
    min_speed: float = 0.0
    target_speed: float = 1.5
    fallback_speed: float = 0.08
    nearest_index_search: int = 10
    course_resolution: float = 0.5
    input_cost: np.ndarray = field(
        default_factory=lambda: np.diag([13.0, 0.0])
    )
    input_change_cost: np.ndarray = field(
        default_factory=lambda: np.diag([26.0, 0.0])
    )
    state_cost: np.ndarray = field(
        default_factory=lambda: np.diag([8.0, 8.0, 1.5, 1.2])
    )
    terminal_state_cost: np.ndarray = field(
        default_factory=lambda: np.diag([15.0, 15.0, 1.2, 2.2])
    )


DEFAULT_MPC_CONFIG = ModelPredictiveControlConfig()
