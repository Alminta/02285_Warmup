import sys
import numpy as np
from abc import ABCMeta, abstractmethod


class Heuristic(metaclass=ABCMeta):
    def __init__(self, initial_state: "State"):
        pass

    def h(self, state: "State") -> "int":

        h, w = len(state.boxes), len(state.boxes[0])

        observed_chars = []

        box_x_pos, box_y_pos = [], []
        goal_x_pos, goal_y_pos = [], []

        # Extract goals & boxes from nested lists
        for x, line in zip(range(h), state.boxes):
            for y, char in zip(range(w), line):
                if char is not None:
                    if char not in observed_chars:
                        observed_chars.append(char)
                        box_x_pos.append([])
                        box_y_pos.append([])
                        goal_x_pos.append([])
                        goal_y_pos.append([])

                    index = observed_chars.index(char)
                    box_x_pos[index].append(x)
                    box_y_pos[index].append(y)

        for x, line in zip(range(h), state.goals):
            for y, char in zip(range(w), line):
                if char is not None:
                    index = observed_chars.index(char.upper())
                    goal_x_pos[index].append(x)
                    goal_y_pos[index].append(y)

        # Convert date to list of numpy arrays
        box_x_pos = [np.asarray(x)[..., np.newaxis] for x in box_x_pos]
        box_y_pos = [np.asarray(x)[..., np.newaxis] for x in box_y_pos]
        goal_x_pos = [np.asarray(x)[..., np.newaxis] for x in goal_x_pos]
        goal_y_pos = [np.asarray(x)[..., np.newaxis] for x in goal_y_pos]

        distance = 0

        uendelig = h + w + 1

        # Calculate distance
        for i in range(len(observed_chars)):
            dist_matrix = np.abs(goal_x_pos[i] - box_x_pos[i].T)
            dist_matrix += np.abs(goal_y_pos[i] - box_y_pos[i].T)

            for _ in range(goal_x_pos[i].shape[0]):
                index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                distance += dist_matrix[index]
                dist_matrix[index[0], ...] = uendelig
                dist_matrix[..., index[1]] = uendelig

        return distance

    @abstractmethod
    def f(self, state: "State") -> "int":
        pass

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class AStar(Heuristic):
    def __init__(self, initial_state: "State"):
        super().__init__(initial_state)

    def f(self, state: "State") -> "int":
        return state.g + self.h(state)

    def __repr__(self):
        return "A* evaluation"


class WAStar(Heuristic):
    def __init__(self, initial_state: "State", w: "int"):
        super().__init__(initial_state)
        self.w = 40

    def f(self, state: "State") -> "int":
        return state.g + self.w * self.h(state)

    def __repr__(self):
        return "WA* ({}) evaluation".format(self.w)


class Greedy(Heuristic):
    def __init__(self, initial_state: "State"):
        super().__init__(initial_state)

    def f(self, state: "State") -> "int":
        return self.h(state)

    def __repr__(self):
        return "Greedy evaluation"

