from copy import deepcopy

from mag_robot_nav.envs.utils import load_config
import scripts.train as train


class FakeVecEnv:
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.closed = False

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self):
        self.learn_calls = []
        self.saved_paths = []

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)

    def save(self, path):
        self.saved_paths.append(path)


def test_seed_parser_supports_ranges_and_deduplicates():
    assert train.parse_seeds("1,3-5,3") == [1, 3, 4, 5]


def test_experiment_passes_n_epochs_and_closes_vec_env(monkeypatch, tmp_path):
    config = deepcopy(load_config("params_baseline/STTRL_DVO/config.yaml"))
    config["training_param"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["training_param"]["tensorboard"] = str(tmp_path / "tensorboard")
    config["training_param"]["env_num"] = 1
    config["training_param"]["time_steps"] = 4

    holder = {}

    def make_vec_env(env_fns):
        holder["vec_env"] = FakeVecEnv(env_fns)
        return holder["vec_env"]

    def make_model(**kwargs):
        holder["ppo_kwargs"] = kwargs
        holder["model"] = FakeModel()
        return holder["model"]

    monkeypatch.setattr(train, "SubprocVecEnv", make_vec_env)
    monkeypatch.setattr(train, "PPO", make_model)

    train.experiment(config)

    assert holder["ppo_kwargs"]["n_epochs"] == config["rl_param"]["n_epochs"]
    assert holder["vec_env"].closed
    assert holder["model"].learn_calls


def test_experiment_closes_vec_env_when_model_creation_fails(monkeypatch, tmp_path):
    config = deepcopy(load_config("params_baseline/STTRL_DVO/config.yaml"))
    config["training_param"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["training_param"]["tensorboard"] = str(tmp_path / "tensorboard")
    config["training_param"]["env_num"] = 1

    holder = {}

    def make_vec_env(env_fns):
        holder["vec_env"] = FakeVecEnv(env_fns)
        return holder["vec_env"]

    def fail_model_creation(**kwargs):
        raise RuntimeError("model init failed")

    monkeypatch.setattr(train, "SubprocVecEnv", make_vec_env)
    monkeypatch.setattr(train, "PPO", fail_model_creation)

    try:
        train.experiment(config)
    except RuntimeError as error:
        assert str(error) == "model init failed"
    else:
        raise AssertionError("experiment() should propagate model init failures")

    assert holder["vec_env"].closed
