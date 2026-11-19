"""Evaluate learned and rule-based navigation methods.

Examples:
    python -m scripts.test --methods sttrl_dvo --seeds 1 --episodes 100
    python -m scripts.test --methods dwa,dstar_mpc,pid_dvo,pid_vo --seeds 1-8
    python -m scripts.test --env maze --render --visualize-attention
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Callable
import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon

from mag_robot_nav.envs.config import (
    EnvType,
    MAX_ANGULAR_VELOCITY,
    MAX_VELOCITY,
    MIN_ANGULAR_VELOCITY,
    MIN_VELOCITY,
)
from mag_robot_nav.envs.geometry import pi_2_pi
from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv
from mag_robot_nav.envs.scenes import build_static_scene
from mag_robot_nav.envs.utils import load_config
from mag_robot_nav.planners.dynamic_window_approach import dwa_control


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "params_baseline" / "STTRL_DVO" / "config.yaml"
DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoint/STTRL_DVO/pretrained/ppo_multi_{model_step}_steps.zip"
)
BASE_CIRCLE_OBSTACLE_COUNT = 10
BASE_CIRCLE_RADIUS_MM = 23.95
BASE_CIRCLE_AREA = math.pi * BASE_CIRCLE_RADIUS_MM ** 2
BASE_OBSTACLE_DENSITY = BASE_CIRCLE_OBSTACLE_COUNT / BASE_CIRCLE_AREA
PID_KP = 0.8
PID_KI = 0.01
PID_KD = 0.001
ENVIRONMENTS = {environment.name: environment for environment in EnvType}


class StoreObstacleDensityRatio(argparse.Action):
    """Store the ratio and remember that the CLI option was explicit."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, "obstacle_density_ratio_explicit", True)


@dataclass(frozen=True)
class MethodSpec:
    """Configuration needed to instantiate one evaluation method."""

    name: str
    controller: str
    config_path: Path = DEFAULT_CONFIG
    using_sampled_vo: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    environment: str
    seed: int
    requested_episodes: int
    completed_episodes: int
    success_rate: float
    mean_episode_speed: float
    mean_episode_timestep: float
    std_episode_timestep: float
    mean_episode_trajectory_length: float
    std_episode_trajectory_length: float


METHODS = {
    "sttrl_dvo": MethodSpec(
        name="STTRL-DVO",
        controller="ppo",
    ),
    "dwa": MethodSpec(name="DWA", controller="dwa"),
    "dstar_mpc": MethodSpec(name="D*-MPC", controller="dstar_mpc"),
    "pid_dvo": MethodSpec(name="PID-DVO", controller="pid"),
    "pid_vo": MethodSpec(
        name="PID-VO",
        controller="pid",
        using_sampled_vo=True,
    ),
}


def normalize_action(linear_velocity: float,
                     angular_velocity: float) -> np.ndarray:
    """Map physical velocity commands to the environment's [-1, 1] action."""
    linear = 2 * (
        (linear_velocity - MIN_VELOCITY)
        / (MAX_VELOCITY - MIN_VELOCITY)
    ) - 1
    angular = 2 * (
        (angular_velocity - MIN_ANGULAR_VELOCITY)
        / (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)
    ) - 1
    return np.clip(np.array([linear, angular], dtype=np.float32), -1, 1)


def nonnegative_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed_value


def nonnegative_float(value: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0:
        raise argparse.ArgumentTypeError(
            "value must be a finite non-negative number"
        )
    return parsed_value


def environment_area(environment: EnvType) -> float:
    """Return the paper-defined area used for obstacle density."""
    if environment == EnvType.circle:
        return BASE_CIRCLE_AREA

    scene = build_static_scene(environment)
    outer_polygons = [
        obstacle for obstacle in scene.obstacles
        if isinstance(obstacle, Polygon)
    ]
    if not outer_polygons:
        raise ValueError(
            f"Environment {environment.name!r} has no boundary Polygon"
        )
    return float(max(outer_polygons, key=lambda polygon: polygon.area).area)


def obstacle_count_from_density(environment: EnvType,
                                density_ratio: float) -> int:
    """Scale the Circle density by ratio and round down as in the paper."""
    count = BASE_OBSTACLE_DENSITY * density_ratio * environment_area(
        environment
    )
    return math.floor(count + 1e-12)


def resolve_obstacle_count(environment: EnvType, obstacle_num: int | None,
                           density_ratio: float,
                           density_ratio_explicit: bool) -> int:
    """Apply CLI precedence between density and absolute obstacle count."""
    if density_ratio_explicit or obstacle_num is None:
        return obstacle_count_from_density(environment, density_ratio)
    return obstacle_num


class Controller:
    """Minimal interface shared by learned and rule-based controllers."""

    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray,
                env: MagRobotNavigationSimEnv) -> np.ndarray:
        raise NotImplementedError


class PPOController(Controller):
    def __init__(self, model_path: Path, device: str,
                 visualize_attention: bool = False):
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device=device)
        self.visualize_attention = bool(visualize_attention)
        self.policy_backbone = self.model.policy.mlp_extractor.pf
        self.policy_backbone.set_attention_capture(self.visualize_attention)

    def predict(self, observation, env):
        action, _ = self.model.predict(observation, deterministic=True)
        if self.visualize_attention:
            env.set_attention_weights(self.policy_backbone.attention_weights)
        return action


class DWAController(Controller):
    def predict(self, observation, env):
        state = np.array([
            env.robot.x, env.robot.y, env.robot.yaw,
            env.robot.vx, env.robot.vw,
        ])
        goal = np.array([env.goal.x, env.goal.y])
        command, trajectory, _ = dwa_control(
            state, env.dwa_config, goal, env.dwa_config.ob)
        env.dwa_predicted_trajectory = trajectory[:, :2]
        return normalize_action(command[0], command[1])


class DStarModelPredictiveControlController(Controller):
    """Adapter around the reusable D*-MPC planner."""

    def __init__(self, verbose: bool = False):
        from mag_robot_nav.planners import DStarModelPredictiveControl

        self.planner = DStarModelPredictiveControl(verbose=verbose)
        self.verbose = bool(verbose)
        self._prediction_count = 0

    def reset(self):
        self.planner.reset()
        self._prediction_count = 0

    def predict(self, observation, env):
        del observation
        started_at = perf_counter()
        action = self.planner.predict(env)
        self._prediction_count += 1
        elapsed = perf_counter() - started_at
        if self.verbose and (self._prediction_count == 1 or elapsed >= 1.0):
            timings = ", ".join(
                f"{name}={seconds:.3f}s"
                for name, seconds in self.planner.last_timings.items()
            )
            print(
                f"  D*-MPC step {self._prediction_count}: {timings}",
                flush=True,
            )
        return action


class PIDGuidanceController(Controller):
    """PID heading controller using the VO/DVO direction as its reference."""

    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def predict(self, observation, env):
        error = float(pi_2_pi(env.d_theta - env.robot.yaw))
        self.integral += error * env.dt
        derivative = (error - self.previous_error) / env.dt
        self.previous_error = error

        angular_velocity = np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative,
            MIN_ANGULAR_VELOCITY,
            MAX_ANGULAR_VELOCITY,
        )
        return normalize_action(0.7 * MAX_VELOCITY, angular_velocity)


def resolve_checkpoint_path(checkpoint_path: Path, seed: int,
                            model_step: int) -> Path:
    """Resolve a checkpoint path/template relative to the project root."""
    try:
        rendered_path = str(checkpoint_path).format(
            seed=seed, model_step=model_step
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "--checkpoint-path supports only {seed} and {model_step} "
            "placeholders"
        ) from error
    resolved_path = Path(rendered_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path.resolve()


def build_controller(spec: MethodSpec, seed: int, model_step: int,
                     device: str,
                     checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
                     visualize_attention: bool = False,
                     verbose: bool = False) -> Controller:
    if spec.controller == "ppo":
        model_path = resolve_checkpoint_path(
            checkpoint_path, seed=seed, model_step=model_step
        )
        return PPOController(
            model_path, device, visualize_attention=visualize_attention
        )
    if spec.controller == "dwa":
        return DWAController()
    if spec.controller == "dstar_mpc":
        return DStarModelPredictiveControlController(verbose=verbose)
    if spec.controller == "pid":
        return PIDGuidanceController()
    raise ValueError(f"Unknown controller type: {spec.controller}")


def evaluate(spec: MethodSpec, seed: int, episodes: int, model_step: int,
             device: str, obstacle_num: int, dynamic_goal: bool,
             render: bool, dvo_heading_mode: str | None,
             checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
             environment: EnvType = EnvType.circle,
             visualize_attention: bool = False,
             verbose: bool = False,
             progress_callback: Callable[[EvaluationResult], None] | None = None,
             ) -> EvaluationResult:
    config = load_config(spec.config_path)
    render = bool(render or visualize_attention)
    config["env"]["cv_show"] = render
    config["env"]["using_tro_vo"] = spec.using_sampled_vo
    if dvo_heading_mode is not None:
        config["env"]["dvo_heading_mode"] = dvo_heading_mode

    base_env = MagRobotNavigationSimEnv(
        config=config,
        evaluation=True,
        save_video=True,
        env_type=environment,
        render_mode="human" if render else None,
    )
    env = gym.wrappers.TimeLimit(
        base_env,
        max_episode_steps=config["env"]["max_episode_steps"],
    )
    controller = build_controller(
        spec, seed, model_step, device, checkpoint_path,
        visualize_attention=visualize_attention,
        verbose=verbose,
    )
    completed_episodes = 0

    try:
        for episode in range(episodes):
            episode_started_at = perf_counter()
            print(
                f"  Episode {episode + 1}/{episodes} started",
                flush=True,
            )
            controller.reset()
            observation, _ = env.reset(
                seed=seed if episode == 0 else None,
                options={
                    "obstacle_num": obstacle_num,
                    "dynamic_goal": dynamic_goal,
                },
            )
            if render:
                # The observation renderer has already submitted the first
                # frame. Pumping the event queue here makes the window visible
                # before a potentially expensive first planning call.
                import cv2

                cv2.waitKey(1)
            terminated = truncated = False

            while not (terminated or truncated):
                action = controller.predict(observation, base_env)
                observation, _, terminated, truncated, _ = env.step(action)
                if render and base_env.cv_render_exit:
                    break

            completed_episodes += 1
            result = _make_result(
                spec, environment, seed, episodes, completed_episodes,
                base_env, verbose=False,
            )
            print(
                f"  Episode {completed_episodes}/{episodes} finished in "
                f"{perf_counter() - episode_started_at:.1f}s; cumulative "
                f"success={result.success_rate:.1%}",
                flush=True,
            )

            if render and base_env.cv_render_exit:
                break

        if progress_callback is not None:
            progress_callback(result)

        return _make_result(
            spec, environment, seed, episodes, completed_episodes,
            base_env, verbose=True,
        )
    finally:
        env.close()


def _make_result(spec: MethodSpec, environment: EnvType, seed: int,
                 requested_episodes: int, completed_episodes: int,
                 env: MagRobotNavigationSimEnv, *,
                 verbose: bool) -> EvaluationResult:
    statistics = env.show_statistics(verbose=verbose)
    return EvaluationResult(
        method=spec.name,
        environment=environment.name,
        seed=seed,
        requested_episodes=requested_episodes,
        completed_episodes=completed_episodes,
        success_rate=float(statistics[0]),
        mean_episode_speed=float(statistics[1]),
        mean_episode_timestep=float(statistics[2]),
        std_episode_timestep=float(statistics[3]),
        mean_episode_trajectory_length=float(statistics[4]),
        std_episode_trajectory_length=float(statistics[5]),
    )


def append_result(csv_path: Path, result: EvaluationResult) -> None:
    """Append a cumulative result, upgrading the legacy CSV schema if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_legacy_result_csv(csv_path, list(asdict(result)))
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(result).keys())
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def _upgrade_legacy_result_csv(csv_path: Path,
                               fieldnames: list[str]) -> None:
    """Add the environment column to result files written by older releases."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] == fieldnames:
        return

    legacy_fields = [name for name in fieldnames if name != "environment"]
    if rows[0] != legacy_fields:
        raise ValueError(
            f"Existing CSV has an incompatible header: {csv_path}"
        )

    migrated_rows = []
    for row in rows[1:]:
        if len(row) == len(legacy_fields):
            row = [row[0], "unknown", *row[1:]]
        elif len(row) != len(fieldnames):
            raise ValueError(
                f"Existing CSV contains an incompatible row: {csv_path}"
            )
        migrated_rows.append(row)

    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(migrated_rows)
    temporary_path.replace(csv_path)


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated seeds and inclusive ranges such as ``1,3-5``."""
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            if end < start:
                raise argparse.ArgumentTypeError("Seed range end must be >= start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def parse_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - METHODS.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")
    return list(dict.fromkeys(methods))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=parse_methods, default=["sttrl_dvo"])
    parser.add_argument("--seeds", type=parse_seeds, default=[1])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model-step", type=int, default=3_000_000)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Checkpoint file or path template relative to the project root; "
            "supports {seed} and {model_step} placeholders."
        ),
    )
    parser.add_argument("--device", default="auto")
    # Obstacle option precedence:
    #   1. An explicitly supplied density ratio always wins.
    #   2. An explicitly supplied obstacle count is used when ratio is omitted.
    #   3. When neither is supplied, use 1x the base Circle density.
    parser.add_argument(
        "--obstacle-num", type=nonnegative_int, default=None,
        help=(
            "Absolute dynamic-obstacle count. Used only when "
            "--obstacle-density-ratio is not explicitly supplied."
        ),
    )
    parser.add_argument(
        "--obstacle-density-ratio",
        type=nonnegative_float,
        action=StoreObstacleDensityRatio,
        default=1.0,
        help=(
            "Multiplier of the base Circle density (10 obstacles in the "
            "Circle; default: 1). If explicitly supplied, this takes "
            "priority over --obstacle-num. Counts are rounded down."
        ),
    )
    parser.set_defaults(obstacle_density_ratio_explicit=False)
    parser.add_argument(
        "--dynamic-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print controller-level timing and fallback diagnostics.",
    )
    parser.add_argument(
        "--visualize-attention", action="store_true",
        help="Render LiDAR patches weighted by STTRL policy attention.",
    )
    parser.add_argument(
        "--env", choices=tuple(ENVIRONMENTS), default=EnvType.circle.name,
        help="Evaluation environment (default: base Circle environment).",
    )
    parser.add_argument(
        "--dvo-heading-mode", choices=("smoothed", "paper"), default="paper")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "algorithm_statistics.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    for method_key in args.methods:
        spec = METHODS[method_key]
        for seed in args.seeds:
            obstacle_num = resolve_obstacle_count(
                ENVIRONMENTS[args.env],
                obstacle_num=args.obstacle_num,
                density_ratio=args.obstacle_density_ratio,
                density_ratio_explicit=args.obstacle_density_ratio_explicit,
            )
            print(f"Evaluating {spec.name}, seed={seed} ...", flush=True)
            print(
                f"  Environment={args.env}, dynamic obstacles={obstacle_num}",
                flush=True,
            )
            result = evaluate(
                spec=spec,
                seed=seed,
                episodes=args.episodes,
                model_step=args.model_step,
                checkpoint_path=args.checkpoint_path,
                device=args.device,
                obstacle_num=obstacle_num,
                dynamic_goal=args.dynamic_goal,
                render=args.render,
                dvo_heading_mode=args.dvo_heading_mode,
                environment=ENVIRONMENTS[args.env],
                visualize_attention=args.visualize_attention,
                verbose=args.verbose,
                progress_callback=lambda partial_result: append_result(
                    args.output, partial_result
                ),
            )
            print(
                f"Saved {result.completed_episodes} cumulative row(s) to "
                f"{args.output}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
