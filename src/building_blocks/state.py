import numpy as np


class State:

    def __init__(self, name, initial_value=0, actions=np.ndarray((0, ))):
        if not isinstance(name, str):
            raise TypeError("Name has to be a string.")
        if not isinstance(initial_value, (int, float)):
            raise TypeError("Value has to be a number.")
        self.name = name
        self.initial_value = initial_value
        self.updated_value = initial_value
        self.actions = actions

    def add_action(self, action):  # Note that it could be a probabilistic action too.
        self.actions = np.append(self.actions, action)

    def update_value(self, value: float):
        self.updated_value = value

    def is_terminal(self):
        return self.actions.shape == (0,)

    def __repr__(self):
        return f'{self.name}: {self.updated_value}\n'
