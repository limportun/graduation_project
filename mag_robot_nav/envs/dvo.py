"""Pure helpers for Dynamic Velocity Obstacle heading selection."""

import numpy as np

from .geometry import pi_2_pi


def merge_angle_intervals(intervals):
    """Merge overlapping linear intervals without mutating the caller's list."""
    merged = []
    for current in sorted((list(interval) for interval in intervals),
                          key=lambda interval: interval[0]):
        if not merged or merged[-1][1] < current[0]:
            merged.append(current)
        else:
            merged[-1][1] = max(merged[-1][1], current[1])
    return merged


def circular_distance(angle_a, angle_b):
    return np.abs(pi_2_pi(np.asarray(angle_a) - angle_b))


def feasible_interval_boundaries(merged_intervals, atol=1e-12):
    if not merged_intervals:
        return np.empty(0, dtype=float)
    boundaries = np.asarray(merged_intervals, dtype=float).reshape(-1)
    wraps = (merged_intervals[0][0] <= -np.pi + atol
             and merged_intervals[-1][1] >= np.pi - atol)
    if wraps:
        boundaries = boundaries[
            ~np.isclose(np.abs(boundaries), np.pi, atol=atol)
        ]
    return np.unique(boundaries) if boundaries.size else boundaries


def select_best_heading(merged_intervals, theta_goal, previous_heading,
                        mode="smoothed", smoothing_tolerance=np.deg2rad(10.0)):
    """Return ``(heading, has_feasible_heading)`` for a DVO policy."""
    if mode not in {"smoothed", "paper"}:
        raise ValueError(f"Unsupported DVO heading mode: {mode!r}")
    theta_goal = float(pi_2_pi(theta_goal))
    if not any(start <= theta_goal <= end
               for start, end in merged_intervals):
        return theta_goal, True

    boundaries = feasible_interval_boundaries(merged_intervals)
    if boundaries.size == 0:
        return float(previous_heading), False

    goal_distances = circular_distance(boundaries, theta_goal)
    minimum = float(np.min(goal_distances))
    if mode == "paper":
        return float(boundaries[int(np.argmin(goal_distances))]), True

    candidates = boundaries[goal_distances <= minimum + smoothing_tolerance]
    previous_distances = circular_distance(candidates, previous_heading)
    return float(candidates[int(np.argmin(previous_distances))]), True
