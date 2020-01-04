import numpy as np


class State:

    def __init__(self, value):
        self.state_count = value
        self.actions = np.array()

    def add_action(self, action):
        self.actions.add(action)
