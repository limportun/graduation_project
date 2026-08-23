"""Minimal interactive example for the model-predictive controller."""

import matplotlib.pyplot as plt

from mag_robot_nav.planners.model_predictive_control import (
    ModelPredictiveControl,
)


def main(show_animation=True, verbose=True):
    path_x = [0.0, 30.0, 6.0, 20.0, 35.0]
    path_y = [0.0, 0.0, 20.0, 35.0, 20.0]
    controller = ModelPredictiveControl(
        -10.0, -10.0, 0.0, 0.0, path_x, path_y, verbose=verbose
    )
    trajectory_x, trajectory_y = [controller.state.x], [controller.state.y]
    for _ in range(500):
        velocity, angular_velocity = controller.plan(
            controller.state.x, controller.state.y,
            controller.state.v, controller.state.yaw,
        )
        from mag_robot_nav.planners.model_predictive_control import update_state
        update_state(controller.state, velocity, angular_velocity,
                     config=controller.config)
        trajectory_x.append(controller.state.x)
        trajectory_y.append(controller.state.y)
        if ((controller.state.x - controller.goal[0]) ** 2
                + (controller.state.y - controller.goal[1]) ** 2) ** 0.5 < 1.5:
            break
    if show_animation:
        plt.plot(controller.cx, controller.cy, "--r", label="reference")
        plt.plot(trajectory_x, trajectory_y, "-b", label="trajectory")
        plt.axis("equal")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
