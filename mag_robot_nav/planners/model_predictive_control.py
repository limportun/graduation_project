"""Iterative linear model-predictive path tracking.

Adapted from PythonRobotics under the MIT License. See
THIRD_PARTY_LICENSES.md for the upstream source and complete notice.

The planner core intentionally contains no plotting or example routes. Runtime
parameters live in :mod:`mpc_config`; see ``examples/mpc_demo.py`` for a demo.
"""

from dataclasses import dataclass
import math
import warnings

import cvxpy
import numpy as np

from . import cubic_spline_planner
from .mpc_config import DEFAULT_MPC_CONFIG, ModelPredictiveControlConfig
from .mpc_utils import constant_speed_profile, smooth_yaw

STATE_DIMENSION = 4  # x, y, velocity, yaw
CONTROL_DIMENSION = 2  # commanded velocity, angular velocity


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0


def pi_2_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_linear_model_matrix(velocity, yaw, angular_velocity,
                            config=DEFAULT_MPC_CONFIG):
    del angular_velocity  # retained in the public signature for compatibility
    matrix_a = np.eye(STATE_DIMENSION)
    matrix_a[0, 2] = config.time_step * math.cos(yaw)
    matrix_a[0, 3] = -config.time_step * velocity * math.sin(yaw)
    matrix_a[1, 2] = config.time_step * math.sin(yaw)
    matrix_a[1, 3] = config.time_step * velocity * math.cos(yaw)

    matrix_b = np.zeros((STATE_DIMENSION, CONTROL_DIMENSION))
    matrix_b[2, 0] = 1.0
    matrix_b[3, 1] = config.time_step

    offset = np.zeros(STATE_DIMENSION)
    offset[0] = config.time_step * velocity * math.sin(yaw) * yaw
    offset[1] = -config.time_step * velocity * math.cos(yaw) * yaw
    return matrix_a, matrix_b, offset


def update_state(state, velocity, angular_velocity,
                 config=DEFAULT_MPC_CONFIG):
    angular_velocity = float(np.clip(
        angular_velocity,
        -config.max_angular_velocity,
        config.max_angular_velocity,
    ))
    state.v = float(np.clip(velocity, config.min_speed, config.max_speed))
    state.x += state.v * math.cos(state.yaw) * config.time_step
    state.y += state.v * math.sin(state.yaw) * config.time_step
    state.yaw += angular_velocity * config.time_step
    return state


def calc_nearest_index(state, course_x, course_y, course_yaw,
                       previous_index, config=DEFAULT_MPC_CONFIG):
    end = min(previous_index + config.nearest_index_search, len(course_x))
    x = np.asarray(course_x[previous_index:end])
    y = np.asarray(course_y[previous_index:end])
    squared_distance = (x - state.x) ** 2 + (y - state.y) ** 2
    local_index = int(np.argmin(squared_distance))
    index = previous_index + local_index
    distance = math.sqrt(float(squared_distance[local_index]))
    direction_error = pi_2_pi(
        course_yaw[index]
        - math.atan2(course_y[index] - state.y, course_x[index] - state.x)
    )
    return index, -distance if direction_error < 0 else distance


def predict_motion(initial_state, velocities, angular_velocities, reference,
                   config=DEFAULT_MPC_CONFIG):
    prediction = np.zeros_like(reference)
    prediction[:, 0] = initial_state
    state = State(
        x=initial_state[0], y=initial_state[1],
        v=initial_state[2], yaw=initial_state[3],
    )
    for index, (velocity, angular_velocity) in enumerate(
            zip(velocities, angular_velocities), start=1):
        update_state(state, velocity, angular_velocity, config=config)
        prediction[:, index] = (state.x, state.y, state.v, state.yaw)
    return prediction


def linear_mpc_control(reference, operating_point, initial_state,
                       angular_reference, *, config=DEFAULT_MPC_CONFIG,
                       verbose=False):
    horizon = config.horizon
    state = cvxpy.Variable((STATE_DIMENSION, horizon + 1))
    control = cvxpy.Variable((CONTROL_DIMENSION, horizon))
    cost = 0.0
    constraints = [state[:, 0] == initial_state]

    for index in range(horizon):
        cost += cvxpy.quad_form(control[:, index], config.input_cost)
        if index:
            cost += cvxpy.quad_form(
                reference[:, index] - state[:, index], config.state_cost
            )
        matrix_a, matrix_b, offset = get_linear_model_matrix(
            operating_point[2, index], operating_point[3, index],
            angular_reference[0, index], config=config,
        )
        constraints.append(
            state[:, index + 1]
            == matrix_a @ state[:, index] + matrix_b @ control[:, index] + offset
        )
        if index < horizon - 1:
            difference = control[:, index + 1] - control[:, index]
            cost += cvxpy.quad_form(difference, config.input_change_cost)
            constraints.append(
                cvxpy.abs(difference[1]) <= config.max_angular_acceleration
            )

    cost += cvxpy.quad_form(
        reference[:, horizon] - state[:, horizon],
        config.terminal_state_cost,
    )
    constraints.extend([
        state[2, :] <= config.max_speed,
        state[2, :] >= config.min_speed,
        cvxpy.abs(control[0, :]) <= config.max_speed,
        cvxpy.abs(control[1, :]) <= config.max_angular_velocity,
    ])
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(solver=cvxpy.ECOS, verbose=verbose)
    if problem.status not in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}:
        warnings.warn(f"MPC solver failed with status {problem.status}",
                      RuntimeWarning)
        return (None,) * 6
    values = (
        control.value[0], control.value[1], state.value[0],
        state.value[1], state.value[3], state.value[2],
    )
    return tuple(np.asarray(value).reshape(-1) for value in values)


def iterative_linear_mpc_control(reference, initial_state, angular_reference,
                                 velocities=None, angular_velocities=None, *,
                                 config=DEFAULT_MPC_CONFIG, verbose=False):
    velocities = ([0.0] * config.horizon if velocities is None else velocities)
    angular_velocities = (
        [0.0] * config.horizon
        if angular_velocities is None else angular_velocities
    )
    result = (None,) * 6
    for _ in range(config.max_iterations):
        operating_point = predict_motion(
            initial_state, velocities, angular_velocities, reference,
            config=config,
        )
        previous_velocity = np.asarray(velocities)
        previous_angular_velocity = np.asarray(angular_velocities)
        result = linear_mpc_control(
            reference, operating_point, initial_state, angular_reference,
            config=config, verbose=verbose,
        )
        velocities, angular_velocities = result[:2]
        if velocities is None:
            break
        change = (np.abs(velocities - previous_velocity).sum()
                  + np.abs(angular_velocities
                           - previous_angular_velocity).sum())
        if change <= config.convergence_threshold:
            break
    else:
        if verbose:
            print("MPC reached the maximum linearization iterations")
    return result


def calc_ref_trajectory(state, course_x, course_y, course_yaw, curvature,
                        speed_profile, resolution, previous_index,
                        config=DEFAULT_MPC_CONFIG):
    del curvature
    reference = np.zeros((STATE_DIMENSION, config.horizon + 1))
    angular_reference = np.zeros((1, config.horizon + 1))
    index, _ = calc_nearest_index(
        state, course_x, course_y, course_yaw, previous_index, config=config
    )
    index = max(index, previous_index)
    travel = 0.0
    for step in range(config.horizon + 1):
        travel += config.target_speed * config.time_step * 5
        course_index = min(index + int(round(travel / resolution)),
                           len(course_x) - 1)
        reference[:, step] = (
            course_x[course_index], course_y[course_index],
            speed_profile[course_index], course_yaw[course_index],
        )
    return reference, index, angular_reference


class ModelPredictiveControl:
    """Stateful MPC path tracker with explicit configuration and verbosity."""

    def __init__(self, x, y, yaw, v, ax, ay, oa=None, odelta=None, *,
                 config: ModelPredictiveControlConfig = None, verbose=False):
        self.config = config or DEFAULT_MPC_CONFIG
        self.verbose = bool(verbose)
        self.oa, self.odelta = oa, odelta
        self.reference = None
        self.state = State(x=x, y=y, yaw=yaw, v=v)
        self._set_course(ax, ay)

    def _set_course(self, path_x, path_y):
        if len(path_x) != len(path_y) or len(path_x) == 0:
            raise ValueError("path_x and path_y must have the same non-zero length")
        if len(path_x) < 3:
            if self.verbose:
                print(f"MPC path expanded from {len(path_x)} control points")
            if len(path_x) == 1:
                path_x = [path_x[0], path_x[0] + 1e-3, path_x[0] + 2e-3]
                path_y = [path_y[0], path_y[0], path_y[0]]
            else:
                path_x = [path_x[0], (path_x[0] + path_x[1]) / 2, path_x[1]]
                path_y = [path_y[0], (path_y[0] + path_y[1]) / 2, path_y[1]]
        self.dl = self.config.course_resolution
        self.cx, self.cy, self.cyaw, self.ck, _ = (
            cubic_spline_planner.calc_spline_course(path_x, path_y, ds=self.dl)
        )
        self.cyaw = smooth_yaw(self.cyaw)
        self.sp = constant_speed_profile(self.cx, self.config.target_speed)
        self.goal = [self.cx[-1], self.cy[-1], self.cyaw[-1]]
        self.state.yaw = self.cyaw[0] + pi_2_pi(self.state.yaw - self.cyaw[0])
        self.target_ind, _ = calc_nearest_index(
            self.state, self.cx, self.cy, self.cyaw, 0, config=self.config
        )

    def update(self, x, y, v, yaw, ax, ay, gx=None, gy=None):
        if len(ax) == 1 and gx is not None and gy is not None:
            ax, ay = [ax[0], (ax[0] + gx) / 2, gx], [ay[0], (ay[0] + gy) / 2, gy]
        self.state = State(x=x, y=y, yaw=yaw, v=v)
        self._set_course(ax, ay)

    def plan(self, x, y, v, yaw):
        self.state.x, self.state.y = x, y
        self.state.v, self.state.yaw = v, yaw
        reference, self.target_ind, angular_reference = calc_ref_trajectory(
            self.state, self.cx, self.cy, self.cyaw, self.ck, self.sp,
            self.dl, self.target_ind, config=self.config,
        )
        self.reference = reference
        initial_state = [x, y, v, yaw]
        result = iterative_linear_mpc_control(
            reference, initial_state, angular_reference,
            self.oa, self.odelta, config=self.config, verbose=self.verbose,
        )
        self.oa, self.odelta = result[:2]
        if self.odelta is None:
            return self.config.fallback_speed, 0.0
        return float(self.oa[0]), float(self.odelta[0])


# Backward-compatible class alias. New code should use the full name.
Mpc = ModelPredictiveControl

__all__ = [
    "ModelPredictiveControl", "ModelPredictiveControlConfig", "Mpc", "State",
    "calc_nearest_index", "calc_ref_trajectory", "get_linear_model_matrix",
    "iterative_linear_mpc_control", "linear_mpc_control", "pi_2_pi",
    "predict_motion", "update_state",
]
