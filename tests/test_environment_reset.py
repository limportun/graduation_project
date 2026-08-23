from pathlib import Path

import numpy as np

from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv
from mag_robot_nav.envs.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_reset_clears_cross_episode_observation_history():
    config = load_config(ROOT / "params_baseline" / "STTRL_DVO" / "config.yaml")
    config["env"]["cv_show"] = False
    config["training_param"]["env_num"] = 1
    env = MagRobotNavigationSimEnv(config=config)
    try:
        env.reset(seed=123)
        env.step(np.zeros(2, dtype=np.float32))
        _, _ = env.reset(seed=124)

        assert len(env.observation_list) == env.observed_time_step_horizon
        first = env.observation_list[0]
        assert all(np.array_equal(first, item) for item in env.observation_list)
        assert len(env.robot_trajectory) == 1
    finally:
        env.close()


def test_reset_options_control_evaluation_scenario():
    config = load_config(ROOT / "params_baseline" / "STTRL_DVO" / "config.yaml")
    config["env"]["cv_show"] = False
    config["training_param"]["env_num"] = 1
    env = MagRobotNavigationSimEnv(config=config, evaluation=True)
    try:
        env.reset(
            seed=321,
            options={"obstacle_num": 0, "dynamic_goal": False},
        )
        assert env.obstacle_num == 0
        assert env.dynamic_goal is False
    finally:
        env.close()
