import numpy as np


class State:

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.actions = np.ndarray((0, ))

    def add_action(self, action):
        self.actions = np.append(self.actions, action)
