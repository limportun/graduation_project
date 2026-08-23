import numpy as np

from mag_robot_nav.envs.config import EnvType
from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv
from mag_robot_nav.envs.utils import load_config
from mag_robot_nav.planners.d_star_model_predictive_control import (
    DStarModelPredictiveControl,
)
import mag_robot_nav.planners.d_star_model_predictive_control as dstar_mpc_module


def test_dstar_mpc_preserves_reference_grid_and_action_semantics():
    config = load_config("params_baseline/STTRL_DVO/config.yaml")
    config["env"]["cv_show"] = False
    environment = MagRobotNavigationSimEnv(
        config=config, evaluation=True, env_type=EnvType.circle
    )
    try:
        environment.reset(
            seed=1,
            options={"obstacle_num": 0, "dynamic_goal": False},
        )
        planner = DStarModelPredictiveControl()

        action = planner.predict(environment)

        assert planner._static_occupancy.shape == (45, 80)
        assert planner.tracker.reference.shape == (4, 3)
        assert environment.planning_path
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all((-1 <= action) & (action <= 1))

        planner.reset()
        assert planner.tracker is None
    finally:
        environment.close()


def test_tracker_path_removes_duplicates_without_random_jitter():
    path_x, path_y = DStarModelPredictiveControl._prepare_tracker_path(
        [1.0, 1.0, 2.0], [3.0, 3.0, 4.0], 2.0, 4.0
    )
    assert path_x == [1.0, 2.0]
    assert path_y == [3.0, 4.0]


def test_dstar_failure_uses_prm_path_in_start_to_goal_order(monkeypatch):
    config = load_config("params_baseline/STTRL_DVO/config.yaml")
    config["env"]["cv_show"] = False
    environment = MagRobotNavigationSimEnv(
        config=config, evaluation=True, env_type=EnvType.circle
    )
    try:
        environment.reset(
            seed=1,
            options={"obstacle_num": 0, "dynamic_goal": False},
        )
        monkeypatch.setattr(
            dstar_mpc_module.Dstar, "run", lambda self, start, end: (None, None)
        )
        monkeypatch.setattr(
            dstar_mpc_module,
            "prm_planning",
            lambda *args, **kwargs: (
                [environment.goal.x, environment.robot.x],
                [environment.goal.y, environment.robot.y],
            ),
        )
        planner = DStarModelPredictiveControl()

        path_x, path_y = planner.plan_path(environment)

        assert planner.last_used_fallback is True
        assert np.isclose(path_x[0], environment.robot.x)
        assert np.isclose(path_y[0], environment.robot.y)
        assert np.isclose(path_x[-1], environment.goal.x)
        assert np.isclose(path_y[-1], environment.goal.y)
    finally:
        environment.close()
