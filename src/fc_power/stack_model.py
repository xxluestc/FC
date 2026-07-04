import numpy as np


class StackMap:
    def __init__(self, current_a, voltage_v):
        self.current = np.asarray(current_a)
        self.voltage = np.asarray(voltage_v)

    def voltage_at(self, current_a):
        return np.interp(current_a, self.current, self.voltage)

    def power_kw_at(self, current_a):
        return np.asarray(current_a) * self.voltage_at(current_a) / 1000
