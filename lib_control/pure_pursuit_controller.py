from dataclasses import dataclass

import numpy as np
import math

# Parameters (keeping these constant)
k = 0.5  # look forward gain
Lfc = 0.05  # [m] look-ahead distance
Kp = 1.0  # speed proportional gain


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    a: float = 0.0  # angular velocity (used in dwa only)

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return (dx**2 + dy**2) ** (1 / 2)  # math.hypot(dx, dy)

    def __str__(self) -> str:
        return f"State(x={self.x}, y={self.y}, yaw={self.yaw}, v={self.v}, a={self.a})"


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


prev_4_pd = 0


def proportional_derivative_control(target, current, Kd: float = 0.1):
    global prev_4_pd
    a = Kp * (target - current) + Kd * (current - prev_4_pd)
    prev_4_pd = current
    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        # self.old_nearest_point_index = None
        self.old_nearest_point_index = 0  # removing redundant computation

    def __len__(self):
        return len(self.cx)

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        elif self.old_nearest_point_index == len(self.cx) - 1:
            # final point (no other choice)
            ind = self.old_nearest_point_index
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                if ind + 1 == len(self.cx):
                    # can't go any further
                    break
                distance_next_index = state.calc_distance(
                    self.cx[ind + 1], self.cy[ind + 1]
                )
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    delta = math.atan2(2.0 * math.sin(alpha) / Lf, 1.0)

    return delta, ind
