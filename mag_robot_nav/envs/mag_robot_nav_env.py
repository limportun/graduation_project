import cv2
import datetime
import math
import os
import warnings
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from typing import Any, Dict, Optional

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points

from mag_robot_nav.envs.config import (
    DT, ROBOT_RADIUS, ROBOT_WIDTH, ROBOT_LENGTH,
    SAFE_DIST, ARRIVAL_THRESHOLD,
    ROI_LENGTH_X, ROI_LENGTH_Y,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    MIN_VELOCITY, MAX_VELOCITY,
    MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    MIN_OBSTACLE_ANGULAR_VELOCITY, MAX_OBSTACLE_ANGULAR_VELOCITY,
    MAX_OBSTACLE_VELOCITY, MAX_OBSTACLE_RADIUS,
    MAX_OBSTACLE_ANGULAR_ACCELERATION,
    EnvType, ObstacleShape, PlanningState,
)
from mag_robot_nav.envs.geometry import pi_2_pi, transform_to_robot_frame
from mag_robot_nav.envs.models import Robot, Goal, Obstacle
from mag_robot_nav.envs.laser import (
    calculate_laser_scan_optimized, compute_rectangle_robot_scan,
)
from mag_robot_nav.envs.utils import (
    generate_random_obstacles, transform_vertices_to_world,
)
from mag_robot_nav.envs.rendering import convert_obstacles_to_image_RGB
from mag_robot_nav.envs.scenes import (
    build_pixel_obstacles, build_static_scene, discretize_static_obstacles,
)
from mag_robot_nav.envs.dvo import (
    circular_distance, feasible_interval_boundaries, merge_angle_intervals,
    select_best_heading,
)
from mag_robot_nav.envs.dwa_config import Config, GoalConfig
from mag_robot_nav.planners.dynamic_window_approach import dwa_control, RobotType
from mag_robot_nav.planners.probabilistic_road_map import prm_planning


# Viridis-approximated BGR color palette for theta-reward interval rendering
_VIRIDIS_BGR = [
    (128, 0, 68),
    (185, 55, 70),
    (231, 127, 62),
    (240, 186, 67),
    (216, 219, 80),
    (166, 226, 116),
    (113, 222, 156),
    (62, 210, 191),
    (35, 192, 218),
    (32, 164, 232),
]


class MagRobotNavigationSimEnv(gym.Env):
    """MagRobot navigation simulation environment with LiDAR-based observations.

    Supports multiple environment types (circle, square, vessel, maze) and
    obstacle shapes (circle, polygon). Uses PRM path planning for large-scale
    environments (vessel, maze) to provide sub-goal guidance.
    """

    def __init__(self, config=None, evaluation=False, save_video=False,
                 env_type=EnvType.circle, obstacle_shape=ObstacleShape.circle,
                 render_mode=None, verbose=None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = DT
        self.robot_radius = ROBOT_RADIUS
        self.robot_width = ROBOT_WIDTH
        self.robot_length = ROBOT_LENGTH
        self.safe_dist = SAFE_DIST
        self.arrival_threshold = ARRIVAL_THRESHOLD
        self.max_episode_steps = config["env"]["max_episode_steps"]
        self.laser_dim = config["network_param"]["lidar_dim"]
        self.laser_max = ROI_LENGTH_Y * config["env"]["laser_max_ratio"]
        self.tank_center = (ROI_LENGTH_X / 2, ROI_LENGTH_Y / 2)
        self.tank_radius = ROI_LENGTH_Y / 2

        self.env_type = env_type
        self.obstacle_shape = obstacle_shape
        self.evaluation = evaluation
        self.verbose = (config["env"].get("verbose", False)
                        if verbose is None else bool(verbose))

        # --- Planning parameters (for vessel/maze large environments) ---
        self.planning_dist_th = ROI_LENGTH_Y * 0.8
        self.sub_goal_index = 3
        self.planning_state = PlanningState.DIRECT
        self._planning_fail_cooldown = 0

        # --- Build static obstacle layout ---
        scene = build_static_scene(self.env_type)
        self.env_width, self.env_length = scene.width, scene.length
        self.static_obstacles = scene.obstacles
        (self.static_circle_obstacles,
         self.static_obstacle_x,
         self.static_obstacle_y) = discretize_static_obstacles(
            self.static_obstacles, self.safe_dist
        )
        self._static_dvo_x = np.fromiter(
            (obstacle.x for obstacle in self.static_circle_obstacles),
            dtype=float,
        )
        self._static_dvo_y = np.fromiter(
            (obstacle.y for obstacle in self.static_circle_obstacles),
            dtype=float,
        )
        self.pixel_static_obstacles = build_pixel_obstacles(
            self.static_obstacles, self.world2pixel
        )

        # --- Config-driven settings ---
        self.using_robot_frame = config["env"]["using_robot_frame"]
        self.env_num = config["training_param"]["env_num"]
        self.using_tro_vo = config["env"]["using_tro_vo"]
        self.dvo_heading_mode = config["env"].get("dvo_heading_mode", "smoothed")
        if self.dvo_heading_mode not in {"smoothed", "paper"}:
            raise ValueError(
                "env.dvo_heading_mode must be either 'smoothed' or 'paper', "
                f"got {self.dvo_heading_mode!r}"
            )
        self.dvo_smoothing_tolerance = np.deg2rad(
            config["env"].get("dvo_smoothing_tolerance_deg", 10.0)
        )

        # --- Randomization capacities ---
        self.training_time_steps = config["training_param"]["time_steps"]
        self.observed_time_step_horizon = config["network_param"]["time_step_horizon"]
        self.capacity_level_2 = self.training_time_steps
        self.capacity_level_3 = self.max_episode_steps

        # --- Domain randomization ---
        self.domain_rand = config["env"]["domain_rand"]
        self.rand_scale = config["env"]["rand_scale"]
        self.x_sigma = self.robot_radius * self.rand_scale
        self.y_sigma = self.robot_radius * self.rand_scale
        self.yaw_sigma = np.pi * self.rand_scale
        self.vx_sigma = MAX_VELOCITY * self.rand_scale * 2
        self.vw_sigma = MAX_ANGULAR_VELOCITY * self.rand_scale * 2

        # --- Curriculum learning ---
        self.config = config
        self.curriculum_learning = self.config["curriculum_learning"]
        self.stage_num = len(self.curriculum_learning["stage_property"])
        self.current_stage = 0
        self.is_static = self.curriculum_learning["stage_property"][self.current_stage][0]
        self.obstacle_num = self.curriculum_learning["stage_property"][self.current_stage][1]
        self.curr_stage_steps_limit = self.curriculum_learning["stage_property"][self.current_stage][2] * 1e3
        self.dynamic_goal = self.curriculum_learning["stage_property"][self.current_stage][3]

        # --- Observation space ---
        self.cv_show = self.config["env"]["cv_show"]
        self.observation_time_step = self.config["network_param"]["time_step"]
        self.image_size = 540
        self.observation_list = []
        self.observation_space = Box(
            -1, 1,
            shape=((7 + self.laser_dim) * self.observation_time_step,)
        )
        self.action_space = Box(-1, 1, shape=(2,))

        # --- Video recording ---
        filepath = "video"
        env_name_dict = {
            EnvType.circle: "circle",
            EnvType.square: "square",
            EnvType.vessel: "vessel",
            EnvType.maze: "maze",
        }
        filename = (
            f"sim_{env_name_dict[self.env_type]}_"
            + datetime.datetime.now().strftime("%m%d-%H%M")
            + ".mp4"
        )
        self.save_video = save_video
        if self.save_video:
            if self.env_type in (EnvType.circle, EnvType.square):
                frame_size = (self.image_size, self.image_size)
            else:
                frame_size = (
                    int(self.env_width / ROI_LENGTH_X * IMAGE_WIDTH),
                    int(self.env_length / ROI_LENGTH_Y * IMAGE_HEIGHT),
                )
            os.makedirs(filepath, exist_ok=True)
            self.sim_video_writer = cv2.VideoWriter(
                os.path.join(filepath, filename),
                cv2.VideoWriter_fourcc(*"mp4v"), 40, frame_size,
            )

        # --- Visualizer ---
        self.rectangle_robot_scan = compute_rectangle_robot_scan(0.0, length=6.0, width=3.0, resolution=1)
        self.robot_trajectory = []
        self.attention_weights = None

        # --- Render canvas dimensions (for OpenCV render) ---
        self._render_w = self.world2pixel(self.env_width)
        self._render_h = self.world2pixel(self.env_length)

        # --- Parameters for collision-free randomization ---
        self.robot_to_goal_safe_threshold = self.robot_radius * 2 + self.safe_dist
        self.static_obstacle_to_robot_safe_threshold = self.robot_radius + self.safe_dist
        self.static_obstacle_to_goal_safe_threshold = self.robot_radius + self.safe_dist
        self.dyn_obstacle_to_robot_safe_threshold = self.robot_radius * 2 + self.safe_dist
        self.dyn_obstacle_to_goal_safe_threshold = self.robot_radius * 2 + self.safe_dist

        # --- Statistics ---
        self.episode_count = 0
        self.success_episode_count = 0
        self.failed_collision_count = 0
        self.failed_out_of_step_limit_count = 0
        self._max_stat_history = 10000  # cap to avoid unbounded memory growth
        self.episode_timesteps_list = []
        self.episode_trajectory_length_list = []
        self.episode_average_speed_list = []
        self.training_timestep = 0
        self._seed = int(config["env"].get("env_seed_base", 0))

    def _log(self, message):
        """Print optional runtime diagnostics without polluting normal training."""
        if self.verbose:
            print(message)

    # =========================================================================
    #  Initialization
    # =========================================================================

    def _sample_robot_goal_pose(self, robot_and_goal_rng):
        """Sample robot and goal poses for the current environment type.

        Returns:
            Tuple of (robot_x, robot_y, robot_yaw, goal_x, goal_y, safe_flag).
        """
        base_mindist = 0.6
        base_maxdist = 0.75
        square_maxdist = 0.9
        large_mindist = 0.05
        large_maxdist = 0.95

        safe_robot_and_goal = True

        if self.env_type == EnvType.circle:
            theta = robot_and_goal_rng.uniform(-np.pi, np.pi)
            robot_to_center = robot_and_goal_rng.uniform(
                self.tank_radius * base_mindist,
                self.tank_radius * base_maxdist)
            robot_x = self.tank_center[0] + robot_to_center * np.cos(theta)
            robot_y = self.tank_center[1] + robot_to_center * np.sin(theta)
            robot_yaw = robot_and_goal_rng.normal(
                np.arctan2(self.tank_center[1] - robot_y,
                           self.tank_center[0] - robot_x), np.pi / 4)
            goal_theta = robot_and_goal_rng.normal(theta + np.pi, np.pi / 4)
            goal_to_center = robot_and_goal_rng.uniform(
                self.tank_radius * base_mindist,
                self.tank_radius * base_maxdist)
            goal_x = self.tank_center[0] + goal_to_center * np.cos(goal_theta)
            goal_y = self.tank_center[1] + goal_to_center * np.sin(goal_theta)

        elif self.env_type == EnvType.square:
            robot_x = robot_and_goal_rng.uniform(
                self.tank_center[0] - self.tank_radius * square_maxdist,
                self.tank_center[0] + self.tank_radius * square_maxdist)
            robot_y = robot_and_goal_rng.uniform(
                self.tank_center[1] - self.tank_radius * square_maxdist,
                self.tank_center[1] + self.tank_radius * square_maxdist)
            robot_yaw = robot_and_goal_rng.normal(
                np.arctan2(self.tank_center[1] - robot_y,
                           self.tank_center[0] - robot_x), np.pi / 4)
            goal_x = robot_and_goal_rng.uniform(
                self.tank_center[0] - self.tank_radius * square_maxdist,
                self.tank_center[0] + self.tank_radius * square_maxdist)
            goal_y = robot_and_goal_rng.uniform(
                self.tank_center[1] - self.tank_radius * square_maxdist,
                self.tank_center[1] + self.tank_radius * square_maxdist)

        elif self.env_type == EnvType.vessel:
            robot_x = robot_and_goal_rng.uniform(
                self.env_width * large_mindist, self.env_width * large_maxdist)
            robot_y = robot_and_goal_rng.uniform(
                self.env_length * large_mindist, self.env_length * large_maxdist)
            robot_yaw = robot_and_goal_rng.uniform(-np.pi, np.pi)
            goal_x = robot_and_goal_rng.uniform(
                self.env_width * large_mindist, self.env_width * large_maxdist)
            goal_y = robot_and_goal_rng.uniform(
                self.env_length * large_mindist, self.env_length * large_maxdist)
            if (not self.static_obstacles[0].contains(Point(robot_x, robot_y))
                    or not self.static_obstacles[0].contains(
                        Point(goal_x, goal_y))):
                safe_robot_and_goal = False

        elif self.env_type == EnvType.maze:
            robot_x = robot_and_goal_rng.uniform(
                self.env_width * large_mindist, self.env_width * large_maxdist)
            robot_y = robot_and_goal_rng.uniform(
                self.env_length * large_mindist, self.env_length * large_maxdist)
            robot_yaw = robot_and_goal_rng.uniform(-np.pi, np.pi)
            goal_x = robot_and_goal_rng.uniform(
                self.env_width * large_mindist, self.env_width * large_maxdist)
            goal_y = robot_and_goal_rng.uniform(
                self.env_length * large_mindist, self.env_length * large_maxdist)
            if (not self.static_obstacles[0].contains(Point(robot_x, robot_y))
                    or not self.static_obstacles[0].contains(
                        Point(goal_x, goal_y))):
                safe_robot_and_goal = False

        return robot_x, robot_y, robot_yaw, goal_x, goal_y, safe_robot_and_goal

    def _collision_free_randomization(self):
        """Generate collision-free robot and goal poses."""
        if self.evaluation:
            self.is_static = False
            base_rho = 10 / 1798.804100059019 # base circle environment 10 obstacles with 1798.804100059019 area
            self.obstacle_num = (
                self.reset_obstacle_num
                if self.reset_obstacle_num is not None
                else np.int32(np.floor(base_rho * self.static_obstacles[0].area))
            )
            self.dynamic_goal = (
                self.reset_dynamic_goal
                if self.reset_dynamic_goal is not None
                else True
            )
        else:
            if (self.training_timestep * self.env_num > self.curr_stage_steps_limit
                    and self.current_stage < self.stage_num - 1):
                self.current_stage += 1
                self.is_static = self.curriculum_learning["stage_property"][self.current_stage][0]
                self.obstacle_num = self.curriculum_learning["stage_property"][self.current_stage][1]
                self.curr_stage_steps_limit = self.curriculum_learning["stage_property"][self.current_stage][2] * 1e3
                self.dynamic_goal = self.curriculum_learning["stage_property"][self.current_stage][3]

        # --- Randomization parameters ---
        maximum_randomization_attempts = 100

        collision_free_randomization = False
        seed_random = False
        while not collision_free_randomization:
            robot_and_goal_rng = np.random.default_rng(
                seed=int(self._seed * self.capacity_level_2 * self.capacity_level_3
                         + self.episode_count * self.capacity_level_3)
            )

            # --- Sample robot & goal ---
            while True:
                robot_x, robot_y, robot_yaw, goal_x, goal_y, safe_robot_and_goal = \
                    self._sample_robot_goal_pose(robot_and_goal_rng)

                # Too close
                if np.hypot(robot_x - goal_x, robot_y - goal_y) < self.robot_to_goal_safe_threshold:
                    safe_robot_and_goal = False

                for static_obs in self.static_obstacles:
                    if isinstance(static_obs, Polygon):
                        dist_to_robot = Point(robot_x, robot_y).distance(static_obs.exterior)
                        dist_to_goal = Point(goal_x, goal_y).distance(static_obs.exterior)
                    else:
                        dist_to_robot = static_obs.distance(Point(robot_x, robot_y))
                        dist_to_goal = static_obs.distance(Point(goal_x, goal_y))
                    if dist_to_robot < self.static_obstacle_to_robot_safe_threshold:
                        safe_robot_and_goal = False
                        break
                    if dist_to_goal < self.static_obstacle_to_goal_safe_threshold:
                        safe_robot_and_goal = False
                        break

                if safe_robot_and_goal:
                    break

            # --- Sample obstacles ---
            obstacle_rng = np.random.default_rng(int(self._seed * self.capacity_level_2 * self.capacity_level_3
                    + self.episode_count * self.capacity_level_3))

            random_times = 0
            while True:
                random_times += 1
                if random_times > maximum_randomization_attempts:
                    if not seed_random:
                        warnings.warn(
                            "Obstacle seeding could not satisfy initialization constraints",
                            RuntimeWarning,
                        )
                    seed_random = True
                    break

                obs_rng = obstacle_rng if not seed_random else None
                self.obstacles = generate_random_obstacles(
                    self.obstacle_num, center=self.tank_center, radius=self.tank_radius,
                    obstacle_rng=obs_rng, env_type=self.env_type,
                    width=self.env_width, length=self.env_length, static_obstacles=self.static_obstacles,
                    obstacle_shape=self.obstacle_shape)

                if self.obstacles is None:
                    continue

                safe = True
                for obs in self.obstacles:
                    if obs.get_distance_to_robot(robot_x, robot_y) < self.dyn_obstacle_to_robot_safe_threshold:
                        safe = False
                        break
                    if obs.get_distance_to_robot(goal_x, goal_y) < self.dyn_obstacle_to_goal_safe_threshold:
                        safe = False
                        break

                if safe:
                    collision_free_randomization = True
                    self._log(f"obstacles seed: {int(self._seed * self.capacity_level_2 * self.capacity_level_3 + self.episode_count * self.capacity_level_3)}, random times: {random_times}")
                    break

        robot = Robot(x=robot_x, y=robot_y, yaw=robot_yaw, vx=0, vy=0, vw=0,
                    delay=self.config["env"]["delay"])
        goal = Goal(x=goal_x, y=goal_y)

        self._log("Finished collision_free_randomization")
        return robot, goal

    # =========================================================================
    #  PRM Path Planning (for large environments: vessel, maze)
    # =========================================================================

    def _sub_goal_planning(self):
        """Plan a path from robot to goal using PRM, return a sub-goal along the path.

        Returns:
            Tuple[Goal, Optional[List[Tuple[float, float]]]]:
                Sub-goal and the full planning path (or None if no path found).
        """
        obstacle_x_list = (self.static_obstacle_x
                           + [obs.x for obs in self.obstacles])
        obstacle_y_list = (self.static_obstacle_y
                           + [obs.y for obs in self.obstacles])

        rx, ry = prm_planning(
            self.robot.x, self.robot.y,
            self.goal.x, self.goal.y,
            obstacle_x_list, obstacle_y_list,
            self.robot_radius + self.safe_dist,
            rng=None, show_animation=False,
            verbose=self.verbose,
        )

        if len(rx) == 0 or len(ry) == 0:
            warnings.warn("PRM found no path; using the goal as sub-goal",
                          RuntimeWarning)
            return self.goal.copy(), None

        path = list(zip(rx, ry))

        # Pick a sub-goal several waypoints ahead from the end
        idx = min(self.sub_goal_index, len(path) - 1)
        sub_x, sub_y = path[-idx - 1]

        self._log(f"[PRM] Path found with {len(path)} waypoints, sub-goal at ({sub_x:.2f}, {sub_y:.2f})")
        return Goal(x=sub_x, y=sub_y), path

    # =========================================================================
    #  Sub-goal state machine (vessel / maze large environments)
    # =========================================================================

    def _update_sub_goal(self):
        """Update the sub-goal for large environments using a PRM-based state machine.

        States:
            DIRECT    — walk straight to goal; trigger PRM when line-of-sight is blocked
            FOLLOWING — advancing along a planned path via intermediate sub-goals
        """
        if self.env_type not in (EnvType.vessel, EnvType.maze):
            self.sub_goal = self.goal.copy()
            return

        # --- Check line-of-sight to goal ---
        robot_goal = LineString([
            (self.robot.x, self.robot.y),
            (self.goal.x, self.goal.y),
        ])
        is_intersection = False
        for obs in self.static_obstacles:
            obs_boundary = obs.boundary if isinstance(obs, Polygon) else obs
            if robot_goal.intersects(obs_boundary):
                is_intersection = True
                break

        dist_to_goal = np.hypot(self.goal.x - self.robot.x,
                                self.goal.y - self.robot.y)

        # --- Decrement planning fail cooldown ---
        if self._planning_fail_cooldown > 0:
            self._planning_fail_cooldown -= 1

        # =====================================================================
        #  State: DIRECT
        # =====================================================================
        if self.planning_state == PlanningState.DIRECT:
            if dist_to_goal < self.planning_dist_th and not is_intersection:
                # Close enough with clear line-of-sight — stay in DIRECT
                self.sub_goal = self.goal.copy()
            elif self._planning_fail_cooldown == 0:
                # Either blocked nearby, or still far from goal — attempt PRM
                if dist_to_goal < self.planning_dist_th:
                    self._log("Start re-planning once!")
                else:
                    self._log("Start init global planning!")
                self._try_planning()
            # else: cooldown active → keep current sub_goal, try again later

        # =====================================================================
        #  State: FOLLOWING
        # =====================================================================
        elif self.planning_state == PlanningState.FOLLOWING:
            dist_to_sub_goal = np.hypot(self.sub_goal.x - self.robot.x,
                                        self.sub_goal.y - self.robot.y)

            if dist_to_goal < self.planning_dist_th and not is_intersection:
                # Clear line-of-sight — cancel plan, go directly to goal
                self.sub_goal = self.goal.copy()
                self.planning_state = PlanningState.DIRECT

            elif dist_to_sub_goal < self.planning_dist_th / 5:
                # Reached current sub-goal — advance to next waypoint
                self.planning_timestep += 1
                idx = min(
                    max(len(self.planning_path)
                        - self.sub_goal_index * self.planning_timestep, 0),
                    len(self.planning_path) - 1,
                )
                self.sub_goal = Goal(
                    x=self.planning_path[idx][0],
                    y=self.planning_path[idx][1])

                if idx == 0:
                    # Reached end of path — re-plan from current position
                    self._log("Start goal re-planning!")
                    self._try_planning()
            # else: keep current sub_goal, still following path

    def _try_planning(self):
        """Attempt PRM planning. On failure, set a cooldown to avoid re-triggering
        every step."""
        self.sub_goal, self.planning_path = self._sub_goal_planning()
        self.planning_timestep = 1

        if self.planning_path is not None:
            self.planning_state = PlanningState.FOLLOWING
            # Sync pixel-coordinate path for rendering
            self.pixel_planning_path = [
                (self.world2pixel(x), self.world2pixel(y))
                for x, y in self.planning_path
            ]
        else:
            # Planning failed — stay in DIRECT, set cooldown to avoid spinning
            if self._planning_fail_cooldown == 0:
                self._planning_fail_cooldown = 50  # ~10 seconds at dt=0.2
            self.planning_state = PlanningState.DIRECT

    # =========================================================================
    #  Coordinate transform
    # =========================================================================

    def world2pixel(self, world_len):
        return int(world_len / ROI_LENGTH_X * IMAGE_WIDTH)

    def _min_max_normalize(self, value, minimum, maximum):
        """Linear transform: [minimum, maximum] -> [-1, 1]."""
        value = np.clip(value, minimum, maximum)
        return ((value - minimum) / (maximum - minimum) - 0.5) * 2

    def _min_max_denormalize(self, normalized_value, minimum, maximum):
        """Linear transform: [-1, 1] -> [minimum, maximum]."""
        normalized_value = np.clip(normalized_value, -1, 1)
        return minimum + (maximum - minimum) * (normalized_value / 2 + 0.5)

    # =========================================================================
    #  Observation
    # =========================================================================

    def _get_observation(self):
        observation_rng = np.random.default_rng(
            seed=int(self._seed * self.capacity_level_2 * self.capacity_level_3
                     + self.episode_count * self.capacity_level_3 + self.timestep))

        unnormalized_observation = np.zeros((7 + self.laser_dim,))

        # --- Observed robot (with optional domain randomization) ---
        self.observed_robot = self.robot.copy()
        self.robot_trajectory.append(
            (self.observed_robot.x, self.observed_robot.y,
             self.observed_robot.yaw, self.observed_robot.vx))
        if self.domain_rand:
            self.observed_robot.x += observation_rng.normal(0, self.x_sigma)
            self.observed_robot.y += observation_rng.normal(0, self.y_sigma)
            self.observed_robot.yaw = pi_2_pi(
                self.observed_robot.yaw + observation_rng.normal(0, self.yaw_sigma))

        # --- Sub-goal planning via PRM (vessel/maze large environments) ---
        self._update_sub_goal()

        # --- Goal in robot frame ---
        relative_goal_pose = transform_to_robot_frame(
            robot_pose=np.array([self.observed_robot.x, self.observed_robot.y,
                                  self.observed_robot.yaw]),
            target_poses=np.expand_dims(
                np.array([self.sub_goal.x, self.sub_goal.y, self.sub_goal.yaw]),
                axis=1),
        )

        unnormalized_observation[0] = self.observed_robot.x
        unnormalized_observation[1] = self.observed_robot.y
        unnormalized_observation[2] = self.observed_robot.yaw
        unnormalized_observation[3] = self.observed_robot.vx
        unnormalized_observation[4] = self.observed_robot.vw
        unnormalized_observation[5] = relative_goal_pose[0]
        unnormalized_observation[6] = relative_goal_pose[1]

        # --- LiDAR scan ---
        self.laser_data = calculate_laser_scan_optimized(
            self.robot.x, self.robot.y, self.robot.yaw,
            self.obstacles, self.laser_max, self.static_obstacles,
            angle_increment=np.deg2rad(360 / self.laser_dim))

        # --- Render for evaluation ---
        if self.evaluation:
            self._render_evaluation_view(relative_goal_pose)

        unnormalized_observation[7:] = self.laser_data

        # --- Normalize ---
        normalized_observation = unnormalized_observation.copy()
        normalized_observation[0] = self._min_max_normalize(normalized_observation[0], 0, ROI_LENGTH_X)
        normalized_observation[1] = self._min_max_normalize(normalized_observation[1], 0, ROI_LENGTH_X)
        normalized_observation[2] = self._min_max_normalize(normalized_observation[2], -np.pi, np.pi)
        normalized_observation[3] = self._min_max_normalize(normalized_observation[3], MIN_VELOCITY, MAX_VELOCITY)
        normalized_observation[4] = self._min_max_normalize(normalized_observation[4], MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        normalized_observation[5] = self._min_max_normalize(normalized_observation[5], -ROI_LENGTH_X, ROI_LENGTH_X)
        normalized_observation[6] = self._min_max_normalize(normalized_observation[6], -ROI_LENGTH_X, ROI_LENGTH_X)
        normalized_observation[7:] = self._min_max_normalize(
            unnormalized_observation[7:], self.robot_radius + self.safe_dist, self.laser_max)

        # --- Build observation history ---
        # Maintain a fixed-size buffer of the most recent observations,
        # then downsample with equal spacing (newest-biased) to produce a
        # compact multi-step input with broad temporal coverage.
        self.observation_list.append(normalized_observation)
        horizon = self.observed_time_step_horizon

        # Pad with the first observation during warm-up
        if len(self.observation_list) < horizon:
            pad = [self.observation_list[0]] * (horizon - len(self.observation_list))
            self.observation_list = pad + self.observation_list
        elif len(self.observation_list) > horizon:
            self.observation_list = self.observation_list[-horizon:]

        # Equally-spaced temporal sampling, newest-biased:
        # indices run from the most recent (horizon-1) backward by 'step'
        # then reverse to chronological order.
        step = horizon // self.observation_time_step
        indices = list(range(horizon - 1, -1, -step))[::-1]
        sampled = [self.observation_list[i] for i in indices]
        normalized_observations = np.concatenate(sampled).astype(
            np.float32, copy=False
        )

        return normalized_observations

    def _render_evaluation_view(self, relative_goal_pose):
        """Render the evaluation visualization with LiDAR, obstacles, and planning path."""
        obstacle_poses = np.array([
            [obstacle.x for obstacle in self.obstacles],
            [obstacle.y for obstacle in self.obstacles],
            [obstacle.yaw for obstacle in self.obstacles],
        ])
        relative_obstacle_poses = transform_to_robot_frame(
            robot_pose=np.array([self.robot.x, self.robot.y, self.robot.yaw]),
            target_poses=obstacle_poses,
        )

        # --- Compute obstacle centres & polygon vertices (both modes) ---
        if self.using_robot_frame:
            center_xs = (relative_obstacle_poses[0, :] / ROI_LENGTH_X * IMAGE_WIDTH
                         + IMAGE_WIDTH / 2).astype(np.int16).copy()
            center_ys = (relative_obstacle_poses[1, :] / ROI_LENGTH_Y * IMAGE_HEIGHT
                         + IMAGE_HEIGHT / 2).astype(np.int16).copy()
        else:
            center_xs = (obstacle_poses[0, :] / ROI_LENGTH_X * IMAGE_WIDTH).astype(np.int16).copy()
            center_ys = (obstacle_poses[1, :] / ROI_LENGTH_Y * IMAGE_HEIGHT).astype(np.int16).copy()

        radii = (np.array([obstacle.radius for obstacle in self.obstacles])
                 / ROI_LENGTH_X * IMAGE_WIDTH).astype(np.int16).copy()

        vertices_list = []
        for obs in self.obstacles:
            if obs.shape == ObstacleShape.circle:
                vertices_list.append(None)
            elif obs.shape == ObstacleShape.polygon:
                vertices_world = transform_vertices_to_world(
                    obs.vertices, obs.x, obs.y, obs.yaw)
                if self.using_robot_frame:
                    # Transform to robot-relative pixel coords (translation + rotation)
                    dx_w = vertices_world[:, 0] - self.robot.x
                    dy_w = vertices_world[:, 1] - self.robot.y
                    cos_r, sin_r = np.cos(self.robot.yaw), np.sin(self.robot.yaw)
                    rel_x = dx_w * cos_r + dy_w * sin_r
                    rel_y = -dx_w * sin_r + dy_w * cos_r
                    vx = rel_x / ROI_LENGTH_X * IMAGE_WIDTH + IMAGE_WIDTH / 2
                    vy = rel_y / ROI_LENGTH_Y * IMAGE_HEIGHT + IMAGE_HEIGHT / 2
                else:
                    vx = vertices_world[:, 0] / ROI_LENGTH_X * IMAGE_WIDTH
                    vy = vertices_world[:, 1] / ROI_LENGTH_Y * IMAGE_HEIGHT
                verts_int = np.vstack((vx.astype(np.int32), vy.astype(np.int32))).T
                vertices_list.append(verts_int.reshape((-1, 1, 2)))

        # --- Call render with mode-specific transforms ---
        if self.using_robot_frame:
            robot_yaw_cos = np.cos(self.robot.yaw)
            robot_yaw_sin = np.sin(self.robot.yaw)

            # Helper: transform world coords -> robot-frame pixel coords
            def _world_to_robot_pixel(wx, wy):
                dx_w = np.asarray(wx) - self.robot.x
                dy_w = np.asarray(wy) - self.robot.y
                rel_x = dx_w * robot_yaw_cos + dy_w * robot_yaw_sin
                rel_y = -dx_w * robot_yaw_sin + dy_w * robot_yaw_cos
                px = rel_x / ROI_LENGTH_X * IMAGE_WIDTH + IMAGE_WIDTH / 2
                py = rel_y / ROI_LENGTH_Y * IMAGE_HEIGHT + IMAGE_HEIGHT / 2
                return px, py

            # Static obstacles (Shapely → pixel → world → robot-frame pixel)
            rf_static_obstacles = []
            for obs in self.pixel_static_obstacles:
                if isinstance(obs, Polygon):
                    coords = np.array(obs.exterior.coords)
                    wx = coords[:, 0] / IMAGE_WIDTH * ROI_LENGTH_X
                    wy = coords[:, 1] / IMAGE_HEIGHT * ROI_LENGTH_Y
                    rpx, rpy = _world_to_robot_pixel(wx, wy)
                    rf_static_obstacles.append(
                        Polygon(np.column_stack([rpx, rpy])))
                elif isinstance(obs, LineString):
                    coords = np.array(obs.coords)
                    wx = coords[:, 0] / IMAGE_WIDTH * ROI_LENGTH_X
                    wy = coords[:, 1] / IMAGE_HEIGHT * ROI_LENGTH_Y
                    rpx, rpy = _world_to_robot_pixel(wx, wy)
                    rf_static_obstacles.append(
                        LineString(np.column_stack([rpx, rpy])))

            # Planned path (world coords → robot-frame pixel)
            rf_planned_path = [
                (int(px), int(py))
                for px, py in zip(*_world_to_robot_pixel(
                    [p[0] for p in self.planning_path],
                    [p[1] for p in self.planning_path]))
            ] if self.planning_path else []

            # Sub-goal (world coords → robot-frame pixel)
            sub_goal_px, sub_goal_py = _world_to_robot_pixel(
                self.sub_goal.x, self.sub_goal.y)

            # Robot trajectory (world coords → robot-frame pixel)
            traj_x, traj_y = _world_to_robot_pixel(
                [t[0] for t in self.robot_trajectory],
                [t[1] for t in self.robot_trajectory])
            rf_traj = [(int(px), int(py), t[2], t[3])
                       for px, py, t in zip(traj_x, traj_y, self.robot_trajectory)]

            map_image = convert_obstacles_to_image_RGB(
                center_xs, center_ys, radii, vertices_list,
                self.world2pixel(self.robot_width),
                self.world2pixel(self.robot_length),
                (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2), 0,
                (int(self.world2pixel(relative_goal_pose[0]) + IMAGE_WIDTH // 2),
                 int(self.world2pixel(relative_goal_pose[1]) + IMAGE_HEIGHT // 2)),
                self.world2pixel(self.arrival_threshold),
                float(relative_goal_pose[2]),
                rf_static_obstacles,
                image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                pixel_planned_path=rf_planned_path,
                pixel_sub_goal=(int(sub_goal_px), int(sub_goal_py)),
                robot_traj=rf_traj)
        else:
            map_image = convert_obstacles_to_image_RGB(
                center_xs, center_ys, radii, vertices_list,
                self.world2pixel(self.robot_width),
                self.world2pixel(self.robot_length),
                (self.world2pixel(self.robot.x), self.world2pixel(self.robot.y)),
                self.robot.yaw,
                (self.world2pixel(self.goal.x), self.world2pixel(self.goal.y)),
                self.world2pixel(self.arrival_threshold), self.goal.yaw,
                self.pixel_static_obstacles,
                image_width=int(self.env_width / ROI_LENGTH_X * IMAGE_WIDTH),
                image_height=int(self.env_length / ROI_LENGTH_Y * IMAGE_HEIGHT),
                pixel_planned_path=self.pixel_planning_path,
                pixel_sub_goal=(self.world2pixel(self.sub_goal.x),
                                self.world2pixel(self.sub_goal.y)),
                robot_traj=[(self.world2pixel(t[0]), self.world2pixel(t[1]),
                             t[2], t[3]) for t in self.robot_trajectory])

        # --- LiDAR scan lines ---
        laser_color = (255, 200, 255)
        laser_thick = 1
        attention_norm_weights = (
            self._get_attention_norm_weights()
            if self.attention_weights is not None
            else None
        )
        for deg_angle, dist in enumerate(self.laser_data):
            world_angle = (self.robot.yaw
                           + np.deg2rad(deg_angle * (360 / self.laser_dim))
                           - np.pi / 2)
            attn_laser_color = laser_color
            if attention_norm_weights is not None:
                # --- Attention weight visualization ---
                num_patches = len(attention_norm_weights)
                attention_index = min(
                    int(deg_angle * num_patches / self.laser_dim),
                    num_patches - 1,
                )
                attn_laser_color = tuple(
                    int(i * attention_norm_weights[attention_index])
                    for i in laser_color
                )
            cv2.line(map_image,
                     (self.world2pixel(self.robot.x + self.rectangle_robot_scan[deg_angle] * np.cos(world_angle)),
                      self.world2pixel(self.robot.y + self.rectangle_robot_scan[deg_angle] * np.sin(world_angle))),
                     (self.world2pixel(self.robot.x + dist * np.cos(world_angle)),
                      self.world2pixel(self.robot.y + dist * np.sin(world_angle))),
                     attn_laser_color, laser_thick)

        # --- DWA predicted trajectory ---
        dwa_pred_traj_color = (255, 255, 0)
        dwa_pred_traj_thick = 2
        for i in range(len(self.dwa_predicted_trajectory) - 1):
            cv2.line(map_image,
                     (self.world2pixel(self.dwa_predicted_trajectory[i, 0]),
                      self.world2pixel(self.dwa_predicted_trajectory[i, 1])),
                     (self.world2pixel(self.dwa_predicted_trajectory[i + 1, 0]),
                      self.world2pixel(self.dwa_predicted_trajectory[i + 1, 1])),
                     dwa_pred_traj_color, dwa_pred_traj_thick)

        # --- Crop and resize ---
        self._render_crop_and_display(map_image)

    def _get_attention_norm_weights(self):
        """Extract normalized attention weights for LiDAR visualization."""
        attention_weights_matrix = self.attention_weights[-1].squeeze(0)
        state_attention_weights = attention_weights_matrix[-1, :]
        tokens_per_step = (
            state_attention_weights.shape[0] // self.observation_time_step
        )
        num_patches = tokens_per_step - 1
        start = (self.observation_time_step - 1) * tokens_per_step
        state_attention_weights = state_attention_weights[
            start:start + num_patches
        ]
        maximum = state_attention_weights.max().clamp_min(
            np.finfo(np.float32).eps
        )
        norm_state_attention_weights = (
            state_attention_weights / maximum
        ).cpu().numpy()
        return np.sqrt(norm_state_attention_weights)

    def _render_crop_and_display(self, map_image):
        """Crop, resize, and display the rendered map image."""
        crop_size = 1080
        start_x = (IMAGE_WIDTH - crop_size) // 2
        start_y = (IMAGE_HEIGHT - crop_size) // 2

        map_image = cv2.flip(map_image, 0)
        if self.env_type in (EnvType.vessel, EnvType.maze):
            cropped_image = map_image.copy()
        else:
            cropped_image = map_image[start_y:start_y + crop_size,
                                      start_x:start_x + crop_size]

        scale = 1.0
        if self.env_type == EnvType.vessel:
            scale = 2.2
        elif self.env_type == EnvType.maze:
            scale = 1.5

        resized_cropped_image = cv2.resize(
            cropped_image,
            (int(scale * self.image_size),
             int(scale * self.image_size * cropped_image.shape[0] / cropped_image.shape[1])))

        if self.cv_show:
            cv2.imshow("map_resized_cropped_image", resized_cropped_image)
            key = cv2.waitKey(int(self.dt * 1000 / 200))
            if key == 27 or key == ord('q') or key == ord('Q'):
                self.cv_render_exit = True

            if self.save_video:
                if self.env_type in (EnvType.circle, EnvType.square):
                    self.sim_video_writer.write(resized_cropped_image)
                elif self.env_type in (EnvType.vessel, EnvType.maze):
                    self.sim_video_writer.write(cropped_image)

    def set_attention_weights(self, attention_weights):
        self.attention_weights = attention_weights

    # =========================================================================
    #  Reward helpers
    # =========================================================================

    def _merge_angle_intervals(self, intervals):
        """Merge collision intervals and retain raw edges for rendering."""
        self.intervals = [edge for interval in intervals for edge in interval]
        return merge_angle_intervals(intervals)

    @staticmethod
    def _circular_distance(angle_a, angle_b):
        return circular_distance(angle_a, angle_b)

    @staticmethod
    def _feasible_interval_boundaries(merged_intervals, atol=1e-12):
        return feasible_interval_boundaries(merged_intervals, atol=atol)

    def _select_best_heading(self, merged_intervals, theta_goal,
                             mode=None, theta_change_threshold=None):
        """Select a guiding heading using the paper or anti-jitter policy.

        ``paper`` implements Algorithm 1: if the goal direction is forbidden,
        choose the feasible boundary with the smallest circular distance.
        ``smoothed`` preserves the original implementation's intent: among
        boundaries within ``minimum distance + tolerance``, prefer the one
        closest to the previously selected heading.

        Returns:
            float: Best heading angle d_theta in radians.
        """
        mode = self.dvo_heading_mode if mode is None else mode
        if theta_change_threshold is None:
            theta_change_threshold = self.dvo_smoothing_tolerance
        heading, feasible = select_best_heading(
            merged_intervals,
            theta_goal,
            previous_heading=float(getattr(self, "d_theta", theta_goal)),
            mode=mode,
            smoothing_tolerance=theta_change_threshold,
        )
        self.dvo_has_feasible_heading = feasible
        return heading

    def _iter_nearby_dvo_obstacles(self, robot_x, robot_y, max_distance):
        """Yield dynamic and nearby discretized static DVO obstacles.

        Static boundaries are immutable, so their coordinates are stored once
        in contiguous arrays.  A vectorized distance mask avoids a Python-level
        scan over the full discretized boundary on every environment step.
        """
        yield from self.obstacles
        if self._static_dvo_x.size == 0:
            return

        distance_squared = (
            (self._static_dvo_x - robot_x) ** 2
            + (self._static_dvo_y - robot_y) ** 2
        )
        nearby_indices = np.flatnonzero(distance_squared < max_distance ** 2)
        for index in nearby_indices:
            yield self.static_circle_obstacles[int(index)]

    def _static_dvo_intervals(self, robot_x, robot_y, max_distance):
        """Build stationary-boundary collision cones with vectorized geometry."""
        if self._static_dvo_x.size == 0:
            return []
        delta_x = self._static_dvo_x - robot_x
        delta_y = self._static_dvo_y - robot_y
        distance_squared = delta_x ** 2 + delta_y ** 2
        mask = distance_squared < max_distance ** 2
        if not np.any(mask):
            return []

        delta_x, delta_y = delta_x[mask], delta_y[mask]
        distances = np.sqrt(distance_squared[mask])
        combined_radius = self.robot_radius + 2 * self.safe_dist
        if np.any(distances <= combined_radius):
            return [[-np.pi, np.pi]]

        theta = np.arctan2(delta_y, delta_x)
        beta = np.arcsin(np.clip(combined_radius / distances, -1.0, 1.0))
        smaller = pi_2_pi(theta - beta)
        bigger = pi_2_pi(theta + beta)
        swap = smaller > bigger
        smaller, bigger = (
            np.where(swap, bigger, smaller),
            np.where(swap, smaller, bigger),
        )
        intervals = []
        for lower, upper, middle in zip(smaller, bigger, theta):
            self._add_interval(lower, upper, middle, intervals)
        return intervals

    def _theta_reward(self, v_x, r_angle=0.001):
        """Our DVO guidance reward function based on velocity-obstacle collision cone analysis.

        Finds the best feasible heading angle d_theta that avoids obstacles while
        staying close to the goal direction. Returns a penalty proportional to
        the deviation from d_theta.

        Args:
            v_x: Forward velocity of the robot.
            r_angle: Scaling factor for the angle penalty.

        Returns:
            float: Guidance reward (typically negative, scaled by r_angle).
        """
        theta_goal = np.arctan2(self.sub_goal.y - self.observed_robot.y,
                                 self.sub_goal.x - self.observed_robot.x)
        dist_to_goal = np.hypot(self.sub_goal.y - self.observed_robot.y,
                                 self.sub_goal.x - self.observed_robot.x)
        self.dangerous_obstacle_dist = np.min([ROI_LENGTH_Y / 3, dist_to_goal])
        if abs(v_x) > 0:
            intervals = self._static_dvo_intervals(
                self.observed_robot.x,
                self.observed_robot.y,
                self.dangerous_obstacle_dist,
            )
            candidate_obstacles = self.obstacles
        else:
            intervals = []
            candidate_obstacles = self._iter_nearby_dvo_obstacles(
                self.observed_robot.x,
                self.observed_robot.y,
                self.dangerous_obstacle_dist,
            )

        for obstacle in candidate_obstacles:
            p_x = obstacle.x - self.observed_robot.x
            p_y = obstacle.y - self.observed_robot.y
            ped_dis = np.hypot(p_x, p_y)
            if ped_dis >= self.dangerous_obstacle_dist:
                continue

            theta = np.arctan2(p_y, p_x)
            combined_radius = self.robot_radius + obstacle.radius + self.safe_dist
            if ped_dis <= combined_radius:
                intervals = [[-np.pi, np.pi]]
                break
            beta = np.arcsin(np.clip(combined_radius / ped_dis, -1.0, 1.0))
            if not np.isfinite(beta):
                continue

            # Obstacle velocity in world frame (vx, vy kinematic model)
            v_bx = obstacle.vx * np.cos(obstacle.yaw) - obstacle.vy * np.sin(obstacle.yaw)
            v_by = obstacle.vx * np.sin(obstacle.yaw) + obstacle.vy * np.cos(obstacle.yaw)

            v_a = abs(v_x)

            # Fast path: static or momentarily stationary obstacle.
            # The collision cone is simply [theta-beta, theta+beta] — no VO solve needed.
            if v_bx == 0.0 and v_by == 0.0 and v_a > 0:
                smaller_angle = pi_2_pi(theta - beta)
                bigger_angle = pi_2_pi(theta + beta)
                if smaller_angle > bigger_angle:
                    smaller_angle, bigger_angle = bigger_angle, smaller_angle
                intervals, _ = self._add_interval(
                    smaller_angle, bigger_angle, theta, intervals)
                continue

            dist_to_circle_center = np.hypot(v_bx, v_by)
            circle_radius = v_a

            if dist_to_circle_center < circle_radius:
                # Case 1: origin inside the velocity circle
                angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta-beta)
                angles += self.solve_angle(v_bx, v_by, v_a, alpha=theta+beta)
                assert len(angles) == 2, (
                    f"len(angles) must be 2 in case 1, got {len(angles)}")
                theta_angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta)
                assert len(theta_angles) == 1, (
                    f"len(theta_angles) must be 1 in case 1, got {len(theta_angles)}")
                theta_angle = theta_angles[0]
                smaller_angle, bigger_angle = min(angles), max(angles)
                intervals, _ = self._add_interval(
                    smaller_angle, bigger_angle, theta_angle, intervals)

            elif np.isclose(dist_to_circle_center, circle_radius,
                            rtol=1e-9, atol=1e-12):
                # Case 2: origin on the velocity circle
                angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta-beta)
                minus_beta_solution_num = len(angles)
                angles += self.solve_angle(v_bx, v_by, v_a, alpha=theta+beta)
                assert len(angles) <= 2, (
                    f"len(angles) must be <= 2 in case 2, got {len(angles)}")

                if len(angles) == 0:
                    pass
                elif len(angles) == 1:
                    theta_ = np.arctan2(-v_by, -v_bx)
                    if abs(theta_ - theta) > np.pi:
                        theta_ += np.sign(theta - theta_) * np.pi * 2

                    if theta_ + np.pi / 2 > theta - beta and theta_ + np.pi / 2 < theta + beta:
                        tangent_alpha = theta_ + np.pi / 2
                    elif theta_ - np.pi / 2 > theta - beta and theta_ - np.pi / 2 < theta + beta:
                        tangent_alpha = theta_ - np.pi / 2
                    else:
                        continue

                    intersection_alpha = (theta - beta if minus_beta_solution_num
                                         else theta + beta)
                    middle_alpha = (tangent_alpha + intersection_alpha) / 2
                    middle_angles = self.solve_angle(v_bx, v_by, v_a, alpha=middle_alpha)
                    assert len(middle_angles) == 1, (
                        f"len(middle_angles) must be 1 in case 2, got {len(middle_angles)}")
                    middle_angle = middle_angles[0]
                    origin_angle = np.arctan2(v_by, v_bx)
                    smaller_angle, bigger_angle = min(angles + [origin_angle]), max(angles + [origin_angle])
                    intervals, _ = self._add_interval(
                        smaller_angle, bigger_angle, middle_angle, intervals)

                elif len(angles) == 2:
                    theta_angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta)
                    assert len(theta_angles) == 1, (
                        f"len(theta_angles) must be 1 in case 2, got {len(theta_angles)}")
                    theta_angle = theta_angles[0]
                    smaller_angle, bigger_angle = min(angles), max(angles)
                    intervals, _ = self._add_interval(
                        smaller_angle, bigger_angle, theta_angle, intervals)

            else:
                # Case 3: origin outside the velocity circle
                angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta-beta)
                angles += self.solve_angle(v_bx, v_by, v_a, alpha=theta+beta)
                assert len(angles) % 2 == 0, (
                    f"len(angles) % 2 must be 0 in case 3, got {len(angles)}")
                assert len(angles) <= 4, (
                    f"len(angles) must be <= 4 in case 3, got {len(angles)}")

                if len(angles) == 0:
                    pass
                elif len(angles) == 2:
                    if angles[0] == angles[1]:
                        continue

                    theta_ = np.arctan2(-v_by, -v_bx)
                    if abs(theta_ - theta) > np.pi:
                        theta_ += np.sign(theta - theta_) * np.pi * 2

                    beta_ = np.arcsin(v_a / np.hypot(v_bx, v_by))
                    assert not np.isnan(beta_)

                    if theta_ + beta_ > theta - beta and theta_ + beta_ < theta + beta:
                        tangent_alpha = theta_ + beta_
                    elif theta_ - beta_ > theta - beta and theta_ - beta_ < theta + beta:
                        tangent_alpha = theta_ - beta_
                    else:
                        continue

                    m = math.tan(tangent_alpha)
                    distance_to_tangent_laser = abs(m * v_bx - v_by) / math.sqrt(m ** 2 + 1)
                    tangent_dist = np.sqrt(max(
                        0.0,
                        v_bx ** 2 + v_by ** 2 - distance_to_tangent_laser ** 2,
                    ))
                    tx = tangent_dist * np.cos(tangent_alpha)
                    ty = tangent_dist * np.sin(tangent_alpha)
                    tangent_angle = np.arctan2(ty + v_by, tx + v_bx)

                    smaller_angle, bigger_angle = min(angles), max(angles)
                    intervals, _ = self._add_interval(
                        smaller_angle, bigger_angle, tangent_angle, intervals)

                elif len(angles) == 4:
                    theta_angles = self.solve_angle(v_bx, v_by, v_a, alpha=theta)
                    assert len(theta_angles) == 2, (
                        f"len(theta_angles) must be 2 in case 3, got {len(theta_angles)}")

                    theta_angle = theta_angles[0]
                    smaller_angle, bigger_angle = min(angles[0], angles[2]), max(angles[0], angles[2])
                    intervals, _ = self._add_interval(
                        smaller_angle, bigger_angle, theta_angle, intervals)

                    theta_angle = theta_angles[1]
                    smaller_angle, bigger_angle = min(angles[1], angles[3]), max(angles[1], angles[3])
                    intervals, _ = self._add_interval(
                        smaller_angle, bigger_angle, theta_angle, intervals)

        # Merge intervals and select best heading
        merged_intervals = self._merge_angle_intervals(intervals)
        d_theta = self._select_best_heading(merged_intervals, theta_goal)

        reward = r_angle * (-np.abs(pi_2_pi(self.observed_robot.yaw - d_theta)) / np.pi)
        self.d_theta = d_theta
        return reward

    def _tro_theta_reward(self, v_x, r_angle=0.001):
        """VO guidance reward using random sampling of feasible headings.

        Returns a penalty proportional to yaw deviation from the best
        collision-free heading direction.
        """
        theta_pre = np.arctan2(self.goal.y - self.robot.y, self.goal.x - self.robot.x)
        d_theta = theta_pre

        self.dangerous_obstacle_dist = ROI_LENGTH_Y / 3
        if self.obstacle_num != 0:
            d_theta = np.pi / 2
            N = 60
            theta_min = 1000
            candidate_obstacles = tuple(self._iter_nearby_dvo_obstacles(
                self.robot.x, self.robot.y, self.dangerous_obstacle_dist))
            for i in range(N):
                theta = np.random.uniform(-np.pi, np.pi)
                free = True
                for obstacle in candidate_obstacles:
                    p_x = obstacle.x - self.robot.x
                    p_y = obstacle.y - self.robot.y
                    ped_dis = np.linalg.norm([p_x, p_y])
                    if ped_dis <= self.dangerous_obstacle_dist:
                        ped_theta = np.arctan2(p_y, p_x)
                        vo_theta = np.arcsin(
                            (self.robot_radius + obstacle.radius + self.safe_dist * 2) / ped_dis)
                        theta_rp = np.arctan2(v_x * np.sin(theta),
                                              v_x * np.cos(theta))
                        if theta_rp >= (ped_theta - vo_theta) and theta_rp <= (ped_theta + vo_theta):
                            free = False
                            break

                if free:
                    theta_diff = (theta - theta_pre) ** 2
                    if theta_diff < theta_min:
                        theta_min = theta_diff
                        d_theta = theta
        else:
            d_theta = theta_pre

        reward = r_angle * (-np.abs(pi_2_pi(self.observed_robot.yaw - d_theta)) / np.pi)
        self.d_theta = d_theta
        return reward

    def solve_angle(self, v_bx, v_by, v_a, alpha):
        """Solve intersection angles between a line and a velocity circle.

        Given obstacle relative velocity (v_bx, v_by), agent speed v_a,
        and line direction alpha, compute the resulting velocity angles
        at intersection points.

        Returns:
            List[float]: Intersection angles in [-pi, pi].
        """
        A = 1
        B = 2 * (v_bx * np.cos(alpha) + v_by * np.sin(alpha))
        C = v_bx ** 2 + v_by ** 2 - v_a ** 2

        discriminant = B ** 2 - 4 * A * C
        intersection_points = []
        if discriminant > -1e-12:
            sqrt_d = np.sqrt(max(0.0, discriminant))
            t1 = (-B - sqrt_d) / (2 * A)
            t2 = (-B + sqrt_d) / (2 * A)
            for t in [t1, t2]:
                if t > 0:
                    x, y = t * np.cos(alpha), t * np.sin(alpha)
                    intersection_points.append((x, y))

        angles = []
        for point in intersection_points:
            angle = np.arctan2(point[1] + v_by, point[0] + v_bx)
            angles.append(angle)
        return angles

    @staticmethod
    def _add_interval(smaller, bigger, middle, intervals):
        """Add a collision interval, handling the wrap-around across ±pi.

        If the middle angle lies between smaller and bigger (counter-clockwise),
        the interval [smaller, bigger] is added directly. Otherwise
        (clockwise/wrap-around), two intervals are added: [bigger, π] and [-π, smaller].

        Returns:
            Tuple[List, bool]: Updated intervals list and across_pi flag.
        """
        if middle > smaller and middle < bigger:
            across_pi = False
            intervals.append([smaller, bigger])
        else:
            across_pi = True
            intervals.append([bigger, np.pi])
            intervals.append([-np.pi, smaller])
        return intervals, across_pi

    # =========================================================================
    #  Reward
    # =========================================================================

    def _get_reward(self):
        reward_parameter = self.config["reward_coefficiency"]
        terminated, truncated = False, False
        collision = False
        safe_dist_to_obstacle = 0.15 * ROI_LENGTH_X

        # --- 1. Tracking reward (goal) ---
        dist_to_goal = np.hypot(self.observed_robot.x - self.goal.x,
                                 self.observed_robot.y - self.goal.y)
        tracking_reward = 0.001 * np.sign(self.last_dist_to_goal - dist_to_goal)
        self.last_dist_to_goal = dist_to_goal

        if dist_to_goal < self.arrival_threshold:
            self._log("[Termination]: Goal!")
            tracking_reward = 1.0
            terminated = True
            self.success_episode_count += 1
            self.success = True

        # --- 2. Collision reward (safety) ---
        min_dist_to_obstacles = np.min(self.laser_data - self.rectangle_robot_scan)

        if min_dist_to_obstacles >= safe_dist_to_obstacle:
            collision_reward = 0.0
        elif min_dist_to_obstacles >= self.safe_dist:
            collision_reward = (
                (min_dist_to_obstacles - safe_dist_to_obstacle)
                / safe_dist_to_obstacle * 0.001
            )
        else:
            collision = True
            self._log("[Truncation]: The robot collided with an obstacle")
            collision_reward = -1.0
            if not terminated:
                self.failed_collision_count += 1

        # --- 3. Guidance reward ---
        if self.using_tro_vo:
            guidance_reward = self._tro_theta_reward(self.robot.vx, r_angle=0.001)
        else:
            guidance_reward = self._theta_reward(self.robot.vx, r_angle=0.001)

        co_guidance_reward = reward_parameter["co_guidance_reward"]

        # --- Total reward ---
        reward = (
            tracking_reward * reward_parameter["co_tracking_reward"]
            + collision_reward * reward_parameter["co_collision_reward"]
            + guidance_reward * co_guidance_reward
        )

        truncated = collision

        # --- Statistics ---
        self.episode_speed_list.append(self.observed_robot.vx)
        self.episode_trajectory_length += np.hypot(
            self.traj_x[-1] - self.traj_x[-2],
            self.traj_y[-1] - self.traj_y[-2])

        if self.timestep >= self.max_episode_steps and not truncated and not terminated:
            self.failed_out_of_step_limit_count += 1

        if terminated:
            self.episode_timesteps_list.append(self.timestep)
            self.episode_trajectory_length_list.append(self.episode_trajectory_length)
            self.episode_average_speed_list.append(np.mean(self.episode_speed_list))
            # Prevent unbounded memory growth during long training runs
            if len(self.episode_timesteps_list) > self._max_stat_history:
                self.episode_timesteps_list = self.episode_timesteps_list[-self._max_stat_history:]
                self.episode_trajectory_length_list = self.episode_trajectory_length_list[-self._max_stat_history:]
                self.episode_average_speed_list = self.episode_average_speed_list[-self._max_stat_history:]

        return reward, terminated, truncated

    # =========================================================================
    #  Gym API
    # =========================================================================

    def set_seed(self, seed=None):
        """Set the environment seed without starting a new episode."""
        self.np_random, seed = seeding.np_random(seed)
        self._seed = int(seed)
        return int(seed)

    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None,
              obstacle_num=None, dynamic_goal=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._seed = int(seed)

        if options:
            obstacle_num = options.get("obstacle_num", obstacle_num)
            dynamic_goal = options.get("dynamic_goal", dynamic_goal)

        self.reset_obstacle_num = obstacle_num
        self.reset_dynamic_goal = dynamic_goal

        self.episode_count += 1
        self.success = False
        self.timestep = 0
        self.episode_trajectory_length = 0
        self.episode_speed_list = []
        self.observation_list = []
        self.robot_trajectory = []

        self.robot, self.goal = self._collision_free_randomization()
        self.dwa_predicted_trajectory = np.zeros((30, 2))
        self.sub_goal = self.goal.copy()
        self.pixel_planning_path = [
            (self.world2pixel(self.sub_goal.x), self.world2pixel(self.sub_goal.y))]
        self.planning_state = PlanningState.DIRECT
        self._planning_fail_cooldown = 0
        self.planning_timestep = 0
        self.planning_path = None
        # Rendering state
        self.traj_x, self.traj_y = [self.robot.x], [self.robot.y]
        self.cv_render_exit = False
        observation = self._get_observation()

        # Initialize guidance direction
        self.d_theta = np.arctan2(self.goal.y - self.robot.y,
                                   self.goal.x - self.robot.x)
        if self.using_tro_vo:
            self._tro_theta_reward(self.robot.vx)
        else:
            self._theta_reward(self.robot.vx)
        self.last_dist_to_goal = np.hypot(self.robot.x - self.goal.x,
                                           self.robot.y - self.goal.y)

        # DWA obstacle configuration
        obs_x = [obs.x for obs in self.obstacles] + self.static_obstacle_x
        obs_y = [obs.y for obs in self.obstacles] + self.static_obstacle_y
        obs_r = [obs.radius for obs in self.obstacles] + [self.safe_dist] * len(self.static_obstacle_x)
        obstacles = np.array([obs_x, obs_y, obs_r]).T
        dist_threshold = ROI_LENGTH_Y / 2
        obstacles_mask = np.hypot(obstacles[:, 0] - self.robot.x,
                                   obstacles[:, 1] - self.robot.y) < dist_threshold
        obstacles = obstacles[obstacles_mask]
        self.dwa_config = Config(obstacles)

        info = {"is_success": self.success}
        return observation, info

    def step(self, normalized_action):
        velocity_rng = np.random.default_rng(
            seed=int(self._seed * self.capacity_level_2 * self.capacity_level_3
                     + self.episode_count * self.capacity_level_3 + self.timestep))

        self.timestep += 1
        self.training_timestep += 1

        # --- Apply action ---
        vx = self._min_max_denormalize(normalized_action[0], MIN_VELOCITY, MAX_VELOCITY)
        vw = self._min_max_denormalize(normalized_action[1], MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        ideal_robot = self.robot.copy()
        ideal_robot.update(target_vx=vx, target_vy=0, target_vw=vw, dt=self.dt)

        if self.domain_rand:
            vx += velocity_rng.normal(0, self.vx_sigma)
            vw += velocity_rng.normal(0, self.vw_sigma)
        self.robot.update(target_vx=vx, target_vy=0, target_vw=vw, dt=self.dt)
        self.robot.vx, self.robot.vw = ideal_robot.vx, ideal_robot.vw
        self.robot.command_queue = ideal_robot.command_queue.copy()

        # --- Update dynamic obstacles ---
        if not self.is_static:
            obs_vel_x = np.clip(velocity_rng.normal(
                MAX_OBSTACLE_VELOCITY / np.sqrt(2) / 2,
                MAX_OBSTACLE_VELOCITY / np.sqrt(2) / 6,
                self.obstacle_num), 0, MAX_OBSTACLE_VELOCITY)
            obs_vel_y = np.clip(velocity_rng.normal(
                MAX_OBSTACLE_VELOCITY / np.sqrt(2) / 2,
                MAX_OBSTACLE_VELOCITY / np.sqrt(2) / 6,
                self.obstacle_num), 0, MAX_OBSTACLE_VELOCITY)
            obs_vel_w = velocity_rng.uniform(
                MIN_OBSTACLE_ANGULAR_VELOCITY, MAX_OBSTACLE_ANGULAR_VELOCITY,
                self.obstacle_num)

            for i, obstacle in enumerate(self.obstacles):
                # Keep obstacles away from goal
                if (np.hypot(self.goal.x - obstacle.x, self.goal.y - obstacle.y)
                        < 2 * self.robot_radius
                        and abs(pi_2_pi(np.arctan2(self.goal.y - obstacle.y,
                                                    self.goal.x - obstacle.x)
                                        - obstacle.yaw)) < np.pi / 2):
                    obstacle.yaw = pi_2_pi(-np.pi + obstacle.yaw)

                # Reflect off static obstacles
                point = Point(obstacle.x, obstacle.y)
                for static_obs in self.static_obstacles:
                    if isinstance(static_obs, Polygon):
                        distance = point.distance(static_obs.exterior)
                    else:
                        distance = static_obs.distance(point)
                    if distance <= MAX_OBSTACLE_RADIUS:
                        if isinstance(static_obs, Polygon):
                            nearest_point = nearest_points(static_obs.boundary, point)[0]
                        else:
                            nearest_point = static_obs.interpolate(
                                static_obs.project(point))

                        normal = np.array([point.x - nearest_point.x,
                                           point.y - nearest_point.y])
                        if abs(pi_2_pi(np.arctan2(normal[1], normal[0])
                                       - obstacle.yaw)) < np.pi / 2:
                            continue
                        normal = normal / np.linalg.norm(normal)
                        vx_world = (obstacle.vx * np.cos(obstacle.yaw)
                                    - obstacle.vy * np.sin(obstacle.yaw))
                        vy_world = (obstacle.vx * np.sin(obstacle.yaw)
                                    + obstacle.vy * np.cos(obstacle.yaw))
                        velocity_vector = np.array([vx_world, vy_world])
                        reflected_vector = (velocity_vector
                                            - 2 * np.dot(velocity_vector, normal) * normal)
                        obstacle.yaw = np.arctan2(reflected_vector[1],
                                                   reflected_vector[0])

                obstacle.update(target_vx=obs_vel_x[i], target_vy=obs_vel_y[i],
                               target_vw=obs_vel_w[i], dt=self.dt)

            # Update DWA obstacle configuration
            obs_x = [obs.x for obs in self.obstacles] + self.static_obstacle_x
            obs_y = [obs.y for obs in self.obstacles] + self.static_obstacle_y
            obs_r = [obs.radius for obs in self.obstacles] + [self.safe_dist] * len(self.static_obstacle_x)
            obstacles = np.array([obs_x, obs_y, obs_r]).T
            dist_threshold = ROI_LENGTH_Y / 2
            obstacles_mask = np.hypot(obstacles[:, 0] - self.robot.x,
                                       obstacles[:, 1] - self.robot.y) < dist_threshold
            obstacles = obstacles[obstacles_mask]
            self.dwa_config = Config(obstacles)

        self.traj_x.append(self.robot.x)
        self.traj_y.append(self.robot.y)

        # --- Dynamic goal (obstacle-aware) ---
        if self.dynamic_goal:
            obs_x = [obs.x for obs in self.obstacles] + self.static_obstacle_x + [self.robot.x]
            obs_y = [obs.y for obs in self.obstacles] + self.static_obstacle_y + [self.robot.y]
            obs_r = ([obs.radius for obs in self.obstacles]
                     + [self.safe_dist] * len(self.static_obstacle_x)
                     + [self.robot_radius])
            goal_obstacles = np.array([obs_x, obs_y, obs_r]).T
            self.goal_dwa_config = GoalConfig(goal_obstacles)
            x = np.array([self.goal.x, self.goal.y, self.goal.yaw,
                          self.goal.vx, self.goal.vw])
            _goal = np.array([ROI_LENGTH_X - self.robot.x, ROI_LENGTH_Y - self.robot.y])
            u, _, min_cost = dwa_control(
                x, self.goal_dwa_config, _goal, self.goal_dwa_config.ob)
            if min_cost == float("Inf"):
                u[0] = 0.2 * self.goal_dwa_config.max_speed
            self.goal.update(target_vx=u[0], target_vy=0.0, target_vw=u[1],
                            dt=self.dt)

        observation = self._get_observation()
        reward, terminated, truncated = self._get_reward()

        info = {"is_success": self.success}
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.save_video:
            self.sim_video_writer.release()
        cv2.destroyAllWindows()
        return super().close()

    # =========================================================================
    #  Statistics & Rendering
    # =========================================================================

    def show_statistics(self, verbose=True):
        self.success_rate = self.success_episode_count / max(self.episode_count, 1)
        self.mean_episode_timestep = np.mean(self.episode_timesteps_list) if self.episode_timesteps_list else 0
        self.std_episode_timestep = np.std(self.episode_timesteps_list) if self.episode_timesteps_list else 0
        self.mean_episode_trajectory_length = np.mean(self.episode_trajectory_length_list) if self.episode_trajectory_length_list else 0
        self.std_episode_trajectory_length = np.std(self.episode_trajectory_length_list) if self.episode_trajectory_length_list else 0
        self.mean_episode_speed = np.mean(self.episode_average_speed_list) if self.episode_average_speed_list else 0

        if verbose:
            print("episode num: %d" % self.episode_count)
            print("episode_success_count: %d" % self.success_episode_count)
            print("failed_collision_count: %d" % self.failed_collision_count)
            print("failed out of step limit count: %d" % self.failed_out_of_step_limit_count)
            print("episode_success_rate: %.2f%%" % (self.success_rate * 100))
            print("episode_timesteps mean: %f, std: %f" % (
                np.mean(self.episode_timesteps_list) if self.episode_timesteps_list else 0,
                np.std(self.episode_timesteps_list) if self.episode_timesteps_list else 0))
            print("episode_trajectory_length mean: %f, std: %f" % (
                np.mean(self.episode_trajectory_length_list) if self.episode_trajectory_length_list else 0,
                np.std(self.episode_trajectory_length_list) if self.episode_trajectory_length_list else 0))
            print("episode_average_speed mean: %f, std: %f" % (
                np.mean(self.episode_average_speed_list) if self.episode_average_speed_list else 0,
                np.std(self.episode_average_speed_list) if self.episode_average_speed_list else 0))

        return (self.success_rate, self.mean_episode_speed,
                self.mean_episode_timestep, self.std_episode_timestep,
                self.mean_episode_trajectory_length, self.std_episode_trajectory_length)

    # =========================================================================
    #  OpenCV render helpers
    # =========================================================================

    def _cv_w2p(self, wx, wy):
        """Convert world coordinates (mm) to pixel coordinates on the render canvas."""
        return (int(wx / ROI_LENGTH_X * self._render_w),
                int(wy / ROI_LENGTH_Y * self._render_h))

    def _cv_draw_tank_edge(self, canvas):
        """Draw the environment boundary (static obstacles) as dark red outlines."""
        for obs in self.pixel_static_obstacles:
            if isinstance(obs, Polygon) and len(obs.exterior.coords) >= 32:
                cx, cy = int(obs.centroid.x), int(obs.centroid.y)
                r = int(np.sqrt(obs.area / np.pi))
                cv2.circle(canvas, (cx, cy), r, color=(0, 0, 150), thickness=8)
            elif isinstance(obs, Polygon):
                coords = np.array(obs.exterior.coords, dtype=np.int32)
                cv2.polylines(canvas, [coords], isClosed=True,
                              color=(0, 0, 150), thickness=8)
            elif isinstance(obs, LineString):
                pt1 = (int(obs.coords[0][0]), int(obs.coords[0][1]))
                pt2 = (int(obs.coords[1][0]), int(obs.coords[1][1]))
                cv2.line(canvas, pt1, pt2, color=(0, 0, 150), thickness=8)

    def _cv_draw_obstacles(self, canvas):
        """Draw dynamic obstacles as filled gray shapes."""
        for obstacle in self.obstacles:
            cx, cy = self._cv_w2p(obstacle.x, obstacle.y)
            r = int(obstacle.radius / ROI_LENGTH_X * self._render_w)
            if obstacle.shape == "circle":
                cv2.circle(canvas, (cx, cy), r, color=(80, 80, 80), thickness=-1)
            elif obstacle.shape == "polygon":
                vertices_world = transform_vertices_to_world(
                    obstacle.vertices, obstacle.x, obstacle.y, obstacle.yaw)
                pixel_vertices = np.array([
                    [int(v[0] / ROI_LENGTH_X * self._render_w),
                     int(v[1] / ROI_LENGTH_Y * self._render_h)]
                    for v in vertices_world
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [pixel_vertices], color=(80, 80, 80))

    def _cv_draw_robot(self, canvas):
        """Draw the robot as a two-color filled rotated rectangle (front red, rear blue)."""
        rx, ry = self._cv_w2p(self.robot.x, self.robot.y)
        rw = int(self.robot_width / ROI_LENGTH_Y * self._render_h)
        rl = int(self.robot_length / ROI_LENGTH_X * self._render_w)
        yaw_deg = np.rad2deg(self.robot.yaw)

        # Front half (red -- direction of motion)
        front_cx = int(rx + rl / 4 * np.cos(self.robot.yaw))
        front_cy = int(ry + rl / 4 * np.sin(self.robot.yaw))
        rect_front = ((front_cx, front_cy), (float(rl) / 2, float(rw)), yaw_deg)
        box_front = cv2.boxPoints(rect_front).astype(np.int32)
        cv2.fillPoly(canvas, [box_front], color=(0, 30, 200))

        # Rear half (blue)
        rear_cx = int(rx - rl / 4 * np.cos(self.robot.yaw))
        rear_cy = int(ry - rl / 4 * np.sin(self.robot.yaw))
        rect_rear = ((rear_cx, rear_cy), (float(rl) / 2, float(rw)), yaw_deg)
        box_rear = cv2.boxPoints(rect_rear).astype(np.int32)
        cv2.fillPoly(canvas, [box_rear], color=(200, 30, 0))

    def _cv_draw_observed_robot(self, canvas):
        """Draw the observed (noisy) robot as a black outline rotated rectangle."""
        rx, ry = self._cv_w2p(self.observed_robot.x, self.observed_robot.y)
        rw = int(self.robot_width / ROI_LENGTH_Y * self._render_h)
        rl = int(self.robot_length / ROI_LENGTH_X * self._render_w)
        yaw_deg = np.rad2deg(self.observed_robot.yaw)
        rect = ((rx, ry), (float(rl), float(rw)), yaw_deg)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(canvas, [box], isClosed=True,
                      color=(0, 0, 0), thickness=2)

    def _cv_draw_goal(self, canvas):
        """Draw the goal as a green star marker with arrival threshold circle."""
        gx, gy = self._cv_w2p(self.goal.x, self.goal.y)
        gr = int(self.arrival_threshold / ROI_LENGTH_X * self._render_w)

        # Arrival threshold circle
        cv2.circle(canvas, (gx, gy), gr, color=(10, 200, 30), thickness=5)
        # Inner filled dot
        cv2.circle(canvas, (gx, gy), 8, color=(0, 150, 0), thickness=-1)

        # Five-pointed star
        star_size = int(self.robot_radius / ROI_LENGTH_X * self._render_w)
        angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, 11)
        radii = np.array([star_size, star_size // 2] * 5 + [star_size])
        pts = np.array([
            [int(gx + np.cos(a) * r), int(gy + np.sin(a) * r)]
            for a, r in zip(angles, radii)
        ], dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                     color=(10, 200, 30), thickness=4)

    def _cv_draw_robot_trajectory(self, canvas):
        """Draw the robot's historical trajectory as a black polyline."""
        if len(self.traj_x) < 2:
            return
        for i in range(len(self.traj_x) - 1):
            pt1 = self._cv_w2p(self.traj_x[i], self.traj_y[i])
            pt2 = self._cv_w2p(self.traj_x[i + 1], self.traj_y[i + 1])
            cv2.line(canvas, pt1, pt2, color=(0, 0, 0), thickness=3)

    def _cv_draw_guidance(self, canvas):
        """Draw the guidance direction (d_theta) as a red line from the robot."""
        if not hasattr(self, 'd_theta'):
            return
        leng = 10.0  # mm
        rx, ry = self._cv_w2p(self.robot.x, self.robot.y)
        ex, ey = self._cv_w2p(self.robot.x + leng * np.cos(self.d_theta),
                               self.robot.y + leng * np.sin(self.d_theta))
        cv2.line(canvas, (rx, ry), (ex, ey), color=(0, 0, 255), thickness=2)

    def _cv_draw_dangerous_circle(self, canvas):
        """Draw the dangerous obstacle distance as a semi-transparent red circle."""
        if not hasattr(self, 'dangerous_obstacle_dist'):
            return
        rx, ry = self._cv_w2p(self.robot.x, self.robot.y)
        rr = int(self.dangerous_obstacle_dist / ROI_LENGTH_X * self._render_w)
        overlay = canvas.copy()
        cv2.circle(overlay, (rx, ry), rr, color=(0, 0, 255), thickness=-1)
        cv2.addWeighted(overlay, 0.05, canvas, 0.95, 0, canvas)

    def _cv_draw_theta_intervals(self, canvas):
        """Draw theta-reward interval boundaries as viridis-colored lines."""
        if self.using_tro_vo or not hasattr(self, 'intervals'):
            return
        leng = 7.5  # mm
        rx, ry = self._cv_w2p(self.robot.x, self.robot.y)
        num_colors = len(_VIRIDIS_BGR)
        for i, edge_angle in enumerate(self.intervals):
            color = _VIRIDIS_BGR[(i // 2) % num_colors]
            ex, ey = self._cv_w2p(
                self.robot.x + leng * np.cos(edge_angle),
                self.robot.y + leng * np.sin(edge_angle))
            cv2.line(canvas, (rx, ry), (ex, ey), color=color, thickness=1)

    def _cv_draw_lidar(self, canvas):
        """Draw LiDAR scan lines as green rays from the robot."""
        if not hasattr(self, 'laser_data'):
            return
        scan_step = max(1, self.laser_dim // 200)
        for deg_angle in range(0, self.laser_dim, scan_step):
            dist = self.laser_data[deg_angle]
            world_angle = (self.robot.yaw
                           + np.deg2rad(deg_angle * (360 / self.laser_dim))
                           - np.pi / 2)
            safe_r = self.rectangle_robot_scan[deg_angle]
            x1, y1 = self._cv_w2p(
                self.robot.x + safe_r * np.cos(world_angle),
                self.robot.y + safe_r * np.sin(world_angle))
            x2, y2 = self._cv_w2p(
                self.robot.x + dist * np.cos(world_angle),
                self.robot.y + dist * np.sin(world_angle))
            cv2.line(canvas, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    def _cv_draw_dwa_trajectory(self, canvas):
        """Draw the DWA predicted trajectory as a blue polyline."""
        if not hasattr(self, 'dwa_predicted_trajectory'):
            return
        traj = self.dwa_predicted_trajectory
        for i in range(len(traj) - 1):
            pt1 = self._cv_w2p(traj[i, 0], traj[i, 1])
            pt2 = self._cv_w2p(traj[i + 1, 0], traj[i + 1, 1])
            cv2.line(canvas, pt1, pt2, color=(255, 0, 0), thickness=2)

    def _cv_draw_text_labels(self, canvas):
        """Draw 'robot' and 'goal' text labels."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        rx, ry = self._cv_w2p(self.robot.x, self.robot.y)
        (tw, th), _ = cv2.getTextSize("robot", font, font_scale, thickness)
        cv2.putText(canvas, "robot", (rx - tw // 2, ry + 20),
                    font, font_scale, (0, 0, 0), thickness)

        gx, gy = self._cv_w2p(self.goal.x, self.goal.y)
        (tw, th), _ = cv2.getTextSize("goal", font, font_scale, thickness)
        cv2.putText(canvas, "goal", (gx - tw // 2, gy + 20),
                    font, font_scale, (0, 0, 0), thickness)

    # =========================================================================
    #  Render (Gymnasium API)
    # =========================================================================

    def render(self, render_mode=None):
        """Render the environment using OpenCV.

        Args:
            render_mode: "human" to display via cv2.imshow, "rgb_array" to return
                         a numpy RGB array. Falls back to self.render_mode if None.

        Returns:
            np.ndarray if render_mode is "rgb_array", None for "human".
        """
        if render_mode is None:
            render_mode = getattr(self, 'render_mode', 'human')

        img_w = self._render_w
        img_h = self._render_h

        # Create canvas (light gray background)
        canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240

        # Draw in layer order (back → front)
        self._cv_draw_tank_edge(canvas)
        self._cv_draw_dangerous_circle(canvas)
        self._cv_draw_obstacles(canvas)
        self._cv_draw_robot_trajectory(canvas)
        self._cv_draw_guidance(canvas)
        self._cv_draw_theta_intervals(canvas)
        self._cv_draw_lidar(canvas)
        self._cv_draw_dwa_trajectory(canvas)
        self._cv_draw_goal(canvas)
        self._cv_draw_robot(canvas)
        self._cv_draw_observed_robot(canvas)
        self._cv_draw_text_labels(canvas)

        # Flip vertically so world-y-up matches image-y-up
        canvas = cv2.flip(canvas, 0)

        # Handle render_mode
        if render_mode == "human" and self.cv_show:
            cv2.imshow("MagRobotNav", canvas)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ord('Q'):
                self.cv_render_exit = True

        if render_mode == "rgb_array":
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        return None
