import argparse
from copy import deepcopy
import os
from pathlib import Path
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces

from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv
from mag_robot_nav.envs.utils import load_config
import torchrl.networks as networks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "params_baseline" / "STTRL_DVO" / "config.yaml"


# Custom Callback to save checkpoints and log metrics
class CustomCheckpointAndLoggingCallback(BaseCallback):
    def __init__(self, save_freq, save_path, algorithm, num_envs, eval_env, num_eval_episodes=1000, verbose=1):
        super(CustomCheckpointAndLoggingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.algorithm = algorithm
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_envs = num_envs  # self.locals.get("rewards").size
        self.curr_episode_cumreward = np.zeros(self.num_envs, dtype=np.float64)
        self.curr_episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self.eval_env = eval_env  # Evaluation environment
        self.num_eval_episodes = num_eval_episodes  # Number of evaluation episodes
        self.success_counts = []

        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Collect reward and length per parallel environment
        envs_reward = self.locals.get("rewards")
        envs_dones = self.locals.get("dones")
        envs_info = self.locals.get("infos")  # Get environment info  dict

        self.curr_episode_cumreward += envs_reward
        self.curr_episode_length += 1

        for i in np.flatnonzero(envs_dones):
            self.success_counts.append(envs_info[i].get("is_success", False))
            self.episode_rewards.append(float(self.curr_episode_cumreward[i]))
            self.episode_lengths.append(int(self.curr_episode_length[i]))
            self.curr_episode_cumreward[i] = 0.0
            self.curr_episode_length[i] = 0

        # Compute success rate every 100 episodes
        if len(self.success_counts) >= 100:
            success_rate = np.mean(self.success_counts)
            self.logger.record("rollout/success_rate", success_rate)
            self.success_counts = []

        # Log mean reward and length every 1000 steps
        if len(self.episode_rewards) >= 1000:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)
            self.logger.dump(self.num_timesteps)
            self.episode_rewards = []
            self.episode_lengths = []

        # Save model
        if self.n_calls * self.num_envs % self.save_freq < self.num_envs:
            current_timesteps = int(self.save_freq) * int(self.n_calls * self.num_envs / self.save_freq)
            checkpoint_path = os.path.join(self.save_path, f"{self.algorithm}_{current_timesteps}_steps")
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {checkpoint_path}")

        return True

    def evaluate_model(self):
        """
        Evaluate model success rate using eval_env.
        """
        success_count = 0
        for _ in range(self.num_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated ,info = self.eval_env.step(action)
                if info.get("is_success", False):  # Based on environment's success flag
                    success_count += 1
                    break
        success_rate = success_count / self.num_eval_episodes
        return success_rate

class PassThroughExtractor(BaseFeaturesExtractor):
    """
    A passthrough features extractor that does nothing and simply returns the input.
    """
    def __init__(self, observation_space, features_dim=None):
        # features_dim computed from observation_space
        if features_dim is None:
            features_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        return observations


class STTRL_DVO_Network(nn.Module):
    """
    Custom network for policy and value function.
    It directly uses the provided `vf` and `pf` networks.
    """

    def __init__(self, config, action_output: int):
        super().__init__()
        self.latent_dim_pi = config["network_param"]["latent_dim_pi"] # 256
        self.latent_dim_vf = config["network_param"]["latent_dim_vf"] # 256
        self.time_step = config["network_param"]["time_step"]
        self.token_dim = config["network_param"]["token_dim"]
        self.using_causal = config["network_param"]["using_causal"]

        # Define shared encoder
        encoder = networks.STTRL_DVO_Encoder(
            _state_dim=config["network_param"]["state_dim"], # 7,
            _lidar_dim=config["network_param"]["lidar_dim"], # 20
            time_step=config["network_param"]["time_step"], # 4,
            token_dim=config["network_param"]["token_dim"] # 32
        )

        # Policy network (policy function, pf)
        self.pf = networks.STTRL_DVO_Backbone(
            encoder=encoder,
            time_step=config["network_param"]["time_step"],
            token_dim=config["network_param"]["token_dim"],
            state_input_shape=config["network_param"]["state_dim"], # 7
            lidar_input_shape=config["network_param"]["lidar_dim"], # 360,
            transformer_params=config["network_param"]["transformer_params"], # [[1, 64], [1, 64]],
            append_hidden_shapes=[[4*self.time_step*self.token_dim,512], [512,self.latent_dim_pi]], # token_dim * sequence_length, next_shape
            using_causal=self.using_causal,
        )

        # Value network (value function, vf)
        self.vf = networks.STTRL_DVO_Backbone(
            encoder=encoder,
            time_step=config["network_param"]["time_step"], # 4,
            token_dim=config["network_param"]["token_dim"], # 32,
            state_input_shape=config["network_param"]["state_dim"], # 7,
            lidar_input_shape=config["network_param"]["lidar_dim"], # 360,
            transformer_params=config["network_param"]["transformer_params"], # [[1, 64], [1, 64]],
            append_hidden_shapes=[[4*self.time_step*self.token_dim,512], [512,self.latent_dim_vf]],
            using_causal=self.using_causal,
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.pf(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.vf(features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through both policy and value networks.

        :param features: The input features for both networks.
        :return: A tuple containing the outputs of the policy and value networks.
        """
        return self.pf(features), self.vf(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        config=None,
        **kwargs,
    ):
        self.config = config
        kwargs["ortho_init"] = False
        kwargs["features_extractor_class"] = PassThroughExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": observation_space.shape[0]}
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Use the custom network as the feature extractor.
        """
        if self.config["training_param"]["architecture"] == "STTRL_DVO":
            self.mlp_extractor = STTRL_DVO_Network(self.config, action_output=self.action_space.shape[0])

def make_env(env_id, rank, config, seed=0):
    def _init():
        def make_mag_robot_navigation_sim_env(**kwargs):
            default_params = {
                "config":config,
            }
            default_params.update(kwargs)
            return MagRobotNavigationSimEnv(**default_params)

        gym.register(
            id = config["training_param"]["env_id"],
            entry_point=make_mag_robot_navigation_sim_env,
            max_episode_steps = config["env"]["max_episode_steps"],
        )

        env = gym.make(env_id)
        env.unwrapped.set_seed(seed + rank)
        return env
    return _init

def linear_schedule(initial_value, verbose=False):
    def func(progress_remaining):
        if verbose:
            print(f"learning rate = {progress_remaining * initial_value:.8f}")
        return progress_remaining * initial_value
    return func

def experiment(config, device="auto"):
    # get parameters
    env_id = config["training_param"]["env_id"]
    time_steps = config["training_param"]["time_steps"]
    algorithm = config["training_param"]["algorithm"]
    num_envs = config["training_param"]["env_num"]
    save_freq = config["training_param"]["save_freq"]
    checkpoint_dir = config["training_param"]["checkpoint_dir"]
    tensorboard = config["training_param"]["tensorboard"]

    checkpoint_callback = CustomCheckpointAndLoggingCallback(save_freq, checkpoint_dir, algorithm, num_envs, eval_env=None)

    policy_kwargs = dict(
        config=config,
    )

    if algorithm != "ppo_multi":
        raise ValueError(f"Unsupported algorithm type: {algorithm}")

    env_seed_base = int(config["env"]["env_seed_base"])
    envs = [make_env(env_id, i, config, seed=env_seed_base)
            for i in range(num_envs)]
    vec_env = None
    try:
        vec_env = SubprocVecEnv(envs)
        model = PPO(
            policy=CustomActorCriticPolicy,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard,
            learning_rate=(
                linear_schedule(
                    config["rl_param"]["learning_rate"],
                    verbose=config["env"].get("verbose", False),
                )
                if config["rl_param"]["using_lr_decay"]
                else config["rl_param"]["learning_rate"]
            ),
            n_steps=config["rl_param"]["n_steps"],
            batch_size=config["rl_param"]["batch_size"],
            n_epochs=config["rl_param"]["n_epochs"],
            gamma=config["rl_param"]["gamma"],
            gae_lambda=config["rl_param"]["gae_lambda"],
            clip_range=config["rl_param"]["clip_range"],
            seed=int(config["rl_param"]["rl_seed"]),
            device=device,
        )

        model.learn(total_timesteps=time_steps, callback=checkpoint_callback)
        final_checkpoint_path = os.path.join(
            checkpoint_dir, f"{algorithm}_{time_steps}_steps")
        model.save(final_checkpoint_path)
    finally:
        if vec_env is not None:
            vec_env.close()

def parse_seeds(value):
    """Parse comma-separated seeds and inclusive ranges such as ``1-8``."""
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            if end < start:
                raise argparse.ArgumentTypeError(
                    "Seed range end must be greater than or equal to start"
                )
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train STTRL-DVO with one or more random seeds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=parse_seeds, default=[1])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    base_config = load_config(args.config)
    for seed in args.seeds:
        config = deepcopy(base_config)
        config["training_param"]["checkpoint_dir"] = os.path.join(
            config["training_param"]["checkpoint_dir"], f"PPO_{seed}"
        )
        config["training_param"]["tensorboard"] = os.path.join(
            config["training_param"]["tensorboard"], f"PPO_{seed}"
        )
        config["rl_param"]["rl_seed"] = seed
        config["env"]["verbose"] = args.verbose
        print(f"============== seed {seed} training start ==============")
        experiment(config, device=args.device)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
