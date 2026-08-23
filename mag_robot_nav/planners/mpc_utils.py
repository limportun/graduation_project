"""Small path-profile helpers shared by the MPC core and examples."""

import math


def constant_speed_profile(course_x, target_speed):
    return [target_speed] * len(course_x)


def smooth_yaw(yaw):
    for index in range(len(yaw) - 1):
        difference = yaw[index + 1] - yaw[index]
        while difference >= math.pi / 2.0:
            yaw[index + 1] -= math.pi * 2.0
            difference = yaw[index + 1] - yaw[index]
        while difference <= -math.pi / 2.0:
            yaw[index + 1] += math.pi * 2.0
            difference = yaw[index + 1] - yaw[index]
    return yaw
