import torch
import numpy as np

from torchrl.networks.base import STTRL_DVO_Encoder
from torchrl.networks.nets import (
    CustomTransformerEncoderLayer,
    STTRL_DVO_Backbone,
)
from mag_robot_nav.envs.config import EnvType
from mag_robot_nav.envs.mag_robot_nav_env import MagRobotNavigationSimEnv
from mag_robot_nav.envs.utils import load_config


def test_causal_mask_is_applied_and_snapshot_is_detached():
    layer = CustomTransformerEncoderLayer(
        d_model=8,
        nhead=1,
        dim_feedforward=16,
        dropout=0,
        batch_first=True,
        capture_attention=True,
    )
    src = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.zeros(4, 4)
    mask.masked_fill_(torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1), float("-inf"))

    layer(src, src_mask=mask)

    assert layer.attn_weights is not None
    assert not layer.attn_weights.requires_grad
    assert torch.count_nonzero(layer.attn_weights[0, 0, 1:]) == 0


def test_training_forward_does_not_retain_attention_weights():
    encoder = STTRL_DVO_Encoder(
        _state_dim=7,
        _lidar_dim=360,
        time_step=3,
        token_dim=32,
    )
    backbone = STTRL_DVO_Backbone(
        encoder=encoder,
        time_step=3,
        token_dim=32,
        state_input_shape=7,
        lidar_input_shape=360,
        transformer_params=((1, 64), (1, 64)),
        append_hidden_shapes=((384, 512), (512, 256)),
        using_causal=True,
    )

    output = backbone(torch.randn(2, 3 * (7 + 360)))

    assert encoder.lidar_patches == 20
    assert output.shape == (2, 256)
    assert backbone.attention_weights == []
    assert all(layer.attn_weights is None for layer in backbone.lidar_append_layers)


def test_attention_capture_can_be_enabled_for_evaluation():
    layer = CustomTransformerEncoderLayer(
        d_model=8,
        nhead=1,
        dim_feedforward=16,
        dropout=0,
        batch_first=True,
    )
    layer(torch.randn(1, 3, 8))
    assert layer.attn_weights is None

    layer.capture_attention = True
    layer(torch.randn(1, 3, 8))
    assert layer.attn_weights is not None
    assert not layer.attn_weights.requires_grad


def test_environment_extracts_current_lidar_patch_attention():
    environment = object.__new__(MagRobotNavigationSimEnv)
    environment.observation_time_step = 3
    weights = torch.zeros(1, 63, 63)
    weights[0, -1, 42:62] = torch.arange(1, 21, dtype=torch.float32)
    environment.attention_weights = [weights]

    normalized = environment._get_attention_norm_weights()

    assert normalized.shape == (20,)
    assert np.isclose(normalized[-1], 1.0)
    assert np.all(np.diff(normalized) > 0)


def test_attention_weights_are_normalized_once_per_render(monkeypatch):
    """Rendering all LiDAR rays must perform only one tensor transfer."""
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
        environment.attention_weights = [torch.ones(1, 63, 63)]
        calls = 0
        original = environment._get_attention_norm_weights

        def count_calls():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(
            environment, "_get_attention_norm_weights", count_calls
        )
        environment._get_observation()

        assert calls == 1
    finally:
        environment.close()
