"""Adapted from PythonRobotics under the MIT License.

Upstream: https://github.com/AtsushiSakai/PythonRobotics
See THIRD_PARTY_LICENSES.md for the complete notice.

D* grid planning

author: Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import heapq
import itertools
import math

from sys import maxsize

import matplotlib.pyplot as plt

import random
import time
import warnings

class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")


class Dstar:
    def __init__(self, maps, *, show_animation=False, verbose=False):
        self.map = maps
        self.open_list = set()
        self._open_heap = []
        self._heap_counter = itertools.count()
        self.show_animation = bool(show_animation)
        self.verbose = bool(verbose)

    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()
        self.remove(x)

        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        self._discard_stale_heap_entries()
        return self._open_heap[0][2] if self._open_heap else None

    def get_kmin(self):
        self._discard_stale_heap_entries()
        return self._open_heap[0][0] if self._open_heap else -1

    def _discard_stale_heap_entries(self):
        """Drop entries superseded by a later decrease-key operation."""
        while self._open_heap:
            key, _, state = self._open_heap[0]
            if state.t == "open" and state in self.open_list and key == state.k:
                break
            heapq.heappop(self._open_heap)

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)
        heapq.heappush(
            self._open_heap, (state.k, next(self._heap_counter), state)
        )

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.discard(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):

        rx = []
        ry = []

        self.insert(end, 0.0)

        # t_s = time.time()
        max_process_iter = 5000
        process_iter = 0
        while True: # when there is no path, this loop will last for too much time !!!
            k_min = self.process_state()
            process_iter += 1
            if start.t == "close":
                break
            if k_min == -1:
                return None, None
            if process_iter >= max_process_iter: # too prevent cost too long time on this loop
                warnings.warn(
                    f"D* reached {max_process_iter} state-processing iterations",
                    RuntimeWarning,
                )
                return None, None
        # print(f"process_iter: {process_iter}")
        # print(f"[run], while -> process_state: {time.time() - t_s} seconds, {1/(time.time() - t_s)} Hz")

        start.set_state("s")
        s = start
        s = s.parent
        if s is None:
            return None, None
        s.set_state("e")
        tmp = start

        # t_s = time.time()
        max_iterations = 1000
        iteration = 0
        visited = set()
        while tmp != end:
            key = (tmp.x, tmp.y)
            if key in visited:
                warnings.warn(
                    "D* path reconstruction encountered a parent cycle",
                    RuntimeWarning,
                )
                return None, None
            visited.add(key)
            if iteration >= max_iterations:
                warnings.warn(
                    f"D* path reconstruction reached {max_iterations} iterations",
                    RuntimeWarning,
                )
                return None, None
            iteration += 1
            # print(f"iteration: {iteration}")

            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            if self.show_animation:
                plt.plot(rx, ry, "-r")
                plt.pause(0.01)
            if tmp.parent is None:
                return None, None
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent

        # print(f"[run], while tmp != end -> iteration: {iteration}, time: {time.time() - t_s} seconds, {1/(time.time() - t_s)} Hz")

        tmp.set_state("e")

        return rx, ry


    def modify(self, state):
        self.modify_cost(state)
        max_iterations = 10
        iteration = 0
        while True:
            k_min = self.process_state()
            # print("i am here !!!")
            if k_min >= state.h:
                break
            if iteration >= max_iterations:
                warnings.warn(
                    f"D* modification reached {max_iterations} iterations",
                    RuntimeWarning,
                )
                break
            iteration += 1

        # print(f"[modify], iteration: {iteration}")


def main(*, show_animation=False, verbose=True):
    m = Map(100, 100)
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)
    # print([(i, j) for i, j in zip(ox, oy)])
    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

    start = [10, 10]
    goal = [50, 50]
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.axis("equal")

    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]
    dstar = Dstar(
        m, show_animation=show_animation, verbose=verbose
    )
    # print(f"start: {start}, end: {end}")
    rx, ry = dstar.run(start, end)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()

def my_main(*, show_animation=False, verbose=True):
    m = Map(100, 100)
    ox, oy = [], []

    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)

    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

    start_pos = [random.randint(0, 50), random.randint(0, 50)]
    goal_pos = [random.randint(0, 50), random.randint(0, 50)]
    start = m.map[start_pos[0]][start_pos[1]]
    end = m.map[goal_pos[0]][goal_pos[1]]

    dstar = Dstar(
        m, show_animation=show_animation, verbose=verbose
    )

    if show_animation:
        plt.figure(figsize=(10, 10))
        plt.plot(ox, oy, ".k")  # static obstacles
        plt.plot(start_pos[0], start_pos[1], "og", label="Start")  # start
        plt.plot(goal_pos[0], goal_pos[1], "xb", label="Goal")  # goal
        plt.legend()
        plt.axis("equal")

    rx, ry = dstar.run(start, end)
    if show_animation:
        path_line, = plt.plot(rx, ry, "-r", label="Path")  # initial path
        plt.pause(0.1)

    for step in range(20):  # run 20 steps of dynamic simulation
        time.sleep(0.5)  # pause 0.5 seconds

        new_start_pos = [
            start_pos[0] + random.randint(-1, 1),
            start_pos[1] + random.randint(-1, 1)
        ]
        if (0 <= new_start_pos[0] < m.row and 0 <= new_start_pos[1] < m.col
            and m.map[new_start_pos[0]][new_start_pos[1]].state != "#"):
            start_pos = new_start_pos

        new_goal_pos = [
            goal_pos[0] + random.randint(-1, 1),
            goal_pos[1] + random.randint(-1, 1)
        ]
        if (0 <= new_goal_pos[0] < m.row and 0 <= new_goal_pos[1] < m.col
            and m.map[new_goal_pos[0]][new_goal_pos[1]].state != "#"):
            goal_pos = new_goal_pos

        start = m.map[start_pos[0]][start_pos[1]]
        end = m.map[goal_pos[0]][goal_pos[1]]

        if random.random() < 1.1:
            new_ox = random.randint(0, 60)
            new_oy = random.randint(0, 60)
            if (new_ox, new_oy) not in zip(ox, oy):
                ox.append(new_ox)
                oy.append(new_oy)
                m.set_obstacle([(new_ox, new_oy)])
                if show_animation:
                    plt.plot(new_ox, new_oy, "xk")  # show new obstacles

        need_replan = False
        if step > 0:  # start checking after first plan
            for x, y in zip(rx, ry):
                if m.map[x][y].state == "#":
                    need_replan = True
                    break

        if need_replan or step == 0:
            dstar.open_list = set()
            dstar.insert(end, 0.0)

            rx, ry = dstar.run(start, end)

            if show_animation:
                path_line.set_xdata(rx)
                path_line.set_ydata(ry)
                plt.plot(start_pos[0], start_pos[1], "og")  # update start position
                plt.plot(goal_pos[0], goal_pos[1], "xb")  # update goal position
                plt.pause(0.1)

        if verbose:
            print(f"Step {step}: Start={start_pos}, Goal={goal_pos}")

    if show_animation:
        plt.show()



if __name__ == '__main__':
    main()
    # my_main()
