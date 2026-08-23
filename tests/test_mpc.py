import numpy as np

from mag_robot_nav.planners.model_predictive_control import (
    ModelPredictiveControl,
)
from mag_robot_nav.planners.mpc_config import ModelPredictiveControlConfig


def test_mpc_uses_explicit_config_and_returns_finite_control():
    config = ModelPredictiveControlConfig(horizon=2, target_speed=1.0)
    controller = ModelPredictiveControl(
        0.0, 0.0, 0.0, 0.0,
        [0.0, 5.0, 10.0], [0.0, 0.0, 0.0],
        config=config,
    )
    control = controller.plan(0.0, 0.0, 0.0, 0.0)
    assert controller.config is config
    assert controller.reference.shape == (4, config.horizon + 1)
    assert np.isfinite(control).all()
