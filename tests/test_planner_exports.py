from pathlib import Path


def test_planner_modules_use_full_names():
    planner_dir = Path(__file__).resolve().parents[1] / "mag_robot_nav" / "planners"
    assert (planner_dir / "model_predictive_control.py").is_file()
    assert (planner_dir / "rapidly_exploring_random_tree.py").is_file()
    assert not (planner_dir / "mpc.py").exists()
    assert not (planner_dir / "rrt.py").exists()


def test_lazy_public_planner_exports():
    from mag_robot_nav.planners import (
        DStarModelPredictiveControl,
        ModelPredictiveControl,
        Mpc,
        RRT,
    )

    assert Mpc is ModelPredictiveControl
    assert RRT.__name__ == "RRT"
    assert DStarModelPredictiveControl.__name__ == (
        "DStarModelPredictiveControl"
    )


def test_dstar_returns_a_path_through_the_available_gap():
    from mag_robot_nav.planners.d_star import Dstar, Map

    grid_map = Map(20, 20)
    grid_map.set_obstacle([(10, y) for y in range(20) if y != 9])
    path_x, path_y = Dstar(grid_map).run(
        grid_map.map[2][2], grid_map.map[17][17]
    )

    assert path_x and path_y
    assert (10, 9) in set(zip(path_x, path_y))
