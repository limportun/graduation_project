"""Path-planning algorithms with lazy public imports.

Importing :mod:`mag_robot_nav.planners` no longer imports matplotlib, cvxpy,
and every planner eagerly. Public symbols remain available from this package.
"""

from importlib import import_module


_EXPORTS = {
    "AStarPlanner": ("a_star", "AStarPlanner"),
    "Dstar": ("d_star", "Dstar"),
    "Map": ("d_star", "Map"),
    "DStarModelPredictiveControl": (
        "d_star_model_predictive_control", "DStarModelPredictiveControl"),
    "Config": ("dynamic_window_approach", "Config"),
    "RobotType": ("dynamic_window_approach", "RobotType"),
    "dwa_control": ("dynamic_window_approach", "dwa_control"),
    "CubicSpline1D": ("cubic_spline_planner", "CubicSpline1D"),
    "CubicSpline2D": ("cubic_spline_planner", "CubicSpline2D"),
    "calc_spline_course": ("cubic_spline_planner", "calc_spline_course"),
    "ModelPredictiveControl": (
        "model_predictive_control", "ModelPredictiveControl"),
    "Mpc": ("model_predictive_control", "Mpc"),
    "ModelPredictiveControlConfig": (
        "mpc_config", "ModelPredictiveControlConfig"),
    "prm_planning": ("probabilistic_road_map", "prm_planning"),
    "RRT": ("rapidly_exploring_random_tree", "RRT"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name):
    """Load a requested planner only when its public symbol is accessed."""
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    module = import_module(f"{__name__}.{module_name}")
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__():
    return sorted((*globals(), *__all__))
