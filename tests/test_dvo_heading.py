import numpy as np
import pytest

from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv


def make_selector(mode="smoothed", previous_heading=0.0, tolerance_deg=10.0):
    env = MagRobotNavigationSimEnv.__new__(MagRobotNavigationSimEnv)
    env.dvo_heading_mode = mode
    env.dvo_smoothing_tolerance = np.deg2rad(tolerance_deg)
    env.d_theta = previous_heading
    return env


def test_paper_mode_chooses_closest_boundary():
    env = make_selector(mode="paper", previous_heading=0.9)
    heading = env._select_best_heading([[-0.5, 0.5]], theta_goal=0.2)
    assert heading == pytest.approx(0.5)
    assert env.dvo_has_feasible_heading


def test_smoothed_mode_prefers_previous_heading_within_tolerance():
    intervals = [[0.0, 1.0]]
    paper = make_selector(mode="paper", previous_heading=0.9)
    smoothed = make_selector(mode="smoothed", previous_heading=0.9)

    assert paper._select_best_heading(intervals, theta_goal=0.45) == pytest.approx(0.0)
    assert smoothed._select_best_heading(intervals, theta_goal=0.45) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("goal", "expected"),
    [(3.0, 2.5), (-3.0, -2.5)],
)
def test_paper_mode_handles_interval_across_pi(goal, expected):
    env = make_selector(mode="paper")
    intervals = [[-np.pi, -2.5], [2.5, np.pi]]
    assert env._select_best_heading(intervals, goal) == pytest.approx(expected)


def test_free_goal_direction_is_unchanged():
    env = make_selector(mode="paper")
    assert env._select_best_heading([[-0.5, 0.5]], 1.0) == pytest.approx(1.0)
    assert env.dvo_has_feasible_heading


def test_fully_blocked_domain_uses_deterministic_fallback():
    env = make_selector(mode="paper", previous_heading=0.75)
    assert env._select_best_heading([[-np.pi, np.pi]], 0.0) == pytest.approx(0.75)
    assert not env.dvo_has_feasible_heading


def test_invalid_heading_mode_is_rejected():
    env = make_selector(mode="paper")
    with pytest.raises(ValueError, match="Unsupported DVO heading mode"):
        env._select_best_heading([[-0.5, 0.5]], 0.0, mode="unknown")


def test_vectorized_static_cones_match_scalar_geometry():
    env = make_selector()
    env._static_dvo_x = np.array([5.0, 0.0, -5.0])
    env._static_dvo_y = np.array([0.0, 5.0, 0.0])
    env.robot_radius = 1.0
    env.safe_dist = 0.5

    vectorized = env._merge_angle_intervals(
        env._static_dvo_intervals(0.0, 0.0, 10.0)
    )
    scalar = []
    for x, y in zip(env._static_dvo_x, env._static_dvo_y):
        distance = np.hypot(x, y)
        theta = np.arctan2(y, x)
        beta = np.arcsin((env.robot_radius + 2 * env.safe_dist) / distance)
        lower = (theta - beta + np.pi) % (2 * np.pi) - np.pi
        upper = (theta + beta + np.pi) % (2 * np.pi) - np.pi
        if lower > upper:
            lower, upper = upper, lower
        env._add_interval(lower, upper, theta, scalar)

    assert np.asarray(vectorized) == pytest.approx(
        np.asarray(env._merge_angle_intervals(scalar))
    )
