import numpy as np


class DegradationCostMap:
    def __init__(self, current_a, cost):
        self.current = np.asarray(current_a)
        self.cost = np.asarray(cost)

    def __call__(self, current_a):
        return np.interp(current_a, self.current, self.cost)
