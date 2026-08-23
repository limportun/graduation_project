import csv

import numpy as np
import pytest

from mag_robot_nav.envs.config import EnvType
from scripts.test import (
    DEFAULT_CHECKPOINT_PATH,
    PROJECT_ROOT,
    DStarModelPredictiveControlController,
    EvaluationResult,
    METHODS,
    PIDGuidanceController,
    PID_KD,
    PID_KI,
    PID_KP,
    append_result,
    build_argument_parser,
    normalize_action,
    obstacle_count_from_density,
    parse_methods,
    parse_seeds,
    resolve_obstacle_count,
    resolve_checkpoint_path,
)


def make_result(seed):
    return EvaluationResult(
        method="STTRL-DVO",
        environment="circle",
        seed=seed,
        requested_episodes=10,
        completed_episodes=10,
        success_rate=0.9,
        mean_episode_speed=1.5,
        mean_episode_timestep=100.0,
        std_episode_timestep=5.0,
        mean_episode_trajectory_length=30.0,
        std_episode_trajectory_length=2.0,
    )


def test_seed_and_method_parsers():
    assert parse_seeds("1,3-5,3") == [1, 3, 4, 5]
    assert parse_methods("sttrl_dvo,dwa,sttrl_dvo") == ["sttrl_dvo", "dwa"]
    with pytest.raises(Exception, match="Unknown methods"):
        parse_methods("missing")


def test_evaluation_cli_defaults_and_environment_selection():
    parser = build_argument_parser()
    defaults = parser.parse_args([])
    assert defaults.env == "circle"
    assert defaults.visualize_attention is False
    assert defaults.checkpoint_path == DEFAULT_CHECKPOINT_PATH
    assert defaults.obstacle_density_ratio == 1.0
    assert defaults.obstacle_density_ratio_explicit is False
    assert defaults.obstacle_num is None

    selected = parser.parse_args([
        "--env", "maze", "--visualize-attention",
        "--methods", "dstar_mpc",
    ])
    assert selected.env == "maze"
    assert selected.visualize_attention is True
    assert selected.methods == ["dstar_mpc"]


def test_obstacle_density_matches_paper_area_rule():
    expected_counts = {
        EnvType.circle: 10,
        EnvType.square: 12,
        EnvType.vessel: 18,
        EnvType.maze: 90,
    }
    assert {
        environment: obstacle_count_from_density(environment, 1.0)
        for environment in EnvType
    } == expected_counts


def test_obstacle_density_has_priority_only_when_explicit():
    assert resolve_obstacle_count(
        EnvType.square, obstacle_num=None, density_ratio=1.0,
        density_ratio_explicit=False,
    ) == 12
    assert resolve_obstacle_count(
        EnvType.square, obstacle_num=7, density_ratio=1.0,
        density_ratio_explicit=False,
    ) == 7
    assert resolve_obstacle_count(
        EnvType.square, obstacle_num=7, density_ratio=0.5,
        density_ratio_explicit=True,
    ) == 6


def test_obstacle_density_cli_marks_explicit_ratio():
    parser = build_argument_parser()
    density_args = parser.parse_args([
        "--obstacle-num", "7", "--obstacle-density-ratio", "0.5",
    ])
    count_args = parser.parse_args(["--obstacle-num", "7"])

    assert density_args.obstacle_density_ratio_explicit is True
    assert count_args.obstacle_density_ratio_explicit is False


def test_checkpoint_path_supports_default_and_seed_templates(tmp_path):
    default_path = resolve_checkpoint_path(
        DEFAULT_CHECKPOINT_PATH, seed=7, model_step=3_000_000
    )
    assert default_path == (
        PROJECT_ROOT / "checkpoint" / "STTRL_DVO" / "pretrained"
        / "ppo_multi_3000000_steps.zip"
    ).resolve()

    explicit_path = resolve_checkpoint_path(
        tmp_path / "PPO_{seed}" / "model_{model_step}.zip",
        seed=2,
        model_step=900_000,
    )
    assert explicit_path == (
        tmp_path / "PPO_2" / "model_900000.zip"
    ).resolve()


def test_result_writer_adds_one_header_and_appends_rows(tmp_path):
    output = tmp_path / "results" / "statistics.csv"
    append_result(output, make_result(seed=1))
    append_result(output, make_result(seed=2))

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row["seed"]) for row in rows] == [1, 2]
    assert rows[0]["method"] == "STTRL-DVO"


def test_result_writer_upgrades_legacy_csv_schema(tmp_path):
    output = tmp_path / "statistics.csv"
    legacy_fields = [
        "method", "seed", "requested_episodes", "completed_episodes",
        "success_rate", "mean_episode_speed", "mean_episode_timestep",
        "std_episode_timestep", "mean_episode_trajectory_length",
        "std_episode_trajectory_length",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(legacy_fields)
        writer.writerow([
            "STTRL-DVO", 1, 10, 10, 0.9, 1.5, 100.0, 5.0, 30.0, 2.0,
        ])

    append_result(output, make_result(seed=2))

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["environment"] == "unknown"
    assert rows[1]["environment"] == "circle"


def test_action_normalization_maps_physical_limits():
    assert np.array_equal(normalize_action(0.0, -np.pi / 4), [-1.0, -1.0])
    assert np.array_equal(normalize_action(2.0, np.pi / 4), [1.0, 1.0])


def test_pid_controller_state_is_reset_between_episodes():
    controller = PIDGuidanceController()
    assert (controller.kp, controller.ki, controller.kd) == (
        PID_KP, PID_KI, PID_KD
    ) == (0.8, 0.01, 0.001)
    controller.integral = 4.0
    controller.previous_error = 2.0
    controller.reset()
    assert controller.integral == 0.0
    assert controller.previous_error == 0.0


def test_method_registry_contains_runnable_controller_types():
    assert {spec.controller for spec in METHODS.values()} == {
        "ppo", "dwa", "dstar_mpc", "pid"
    }
    assert isinstance(DStarModelPredictiveControlController(),
                      DStarModelPredictiveControlController)
