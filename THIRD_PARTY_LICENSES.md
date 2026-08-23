# Third-party software notices

This repository contains modified implementations from the
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) project.
PythonRobotics is Copyright (c) 2016-present Atsushi Sakai and other
contributors and is distributed under the MIT License.

## Adapted files

The following paths retain their original author headers where present and
have been modified for this project, including API normalization, verbosity
control, lazy imports, numerical safeguards, and integration with STTRL-DVO.

| STTRL-DVO path | PythonRobotics source |
| --- | --- |
| `mag_robot_nav/planners/a_star.py` | [`PathPlanning/AStar/a_star.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py) |
| `mag_robot_nav/planners/cubic_spline_planner.py` | [`PathPlanning/CubicSpline/cubic_spline_planner.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/CubicSpline/cubic_spline_planner.py) |
| `mag_robot_nav/planners/d_star.py` | [`PathPlanning/DStar/dstar.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DStar/dstar.py) |
| `mag_robot_nav/planners/dynamic_window_approach.py` | [`PathPlanning/DynamicWindowApproach/dynamic_window_approach.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py) |
| `mag_robot_nav/planners/probabilistic_road_map.py` | [`PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py) |
| `mag_robot_nav/planners/rapidly_exploring_random_tree.py` | [`PathPlanning/RRT/rrt.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py) |
| `mag_robot_nav/planners/model_predictive_control.py` and related MPC helpers | [`PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py`](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py) |

Upstream repository: https://github.com/AtsushiSakai/PythonRobotics

Upstream license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE

## PythonRobotics MIT License

```text
The MIT License (MIT)

Copyright (c) 2016 - now Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
