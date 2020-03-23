import numpy as np


class State:

    def __init__(self, name, value=0):
        if not isinstance(name, str):
            raise TypeError("Name has to be a string.")
        if not isinstance(value, (int, float)):
            raise TypeError("Value has to be a number.")
        self.name = name
        self.value = value
        self.actions = np.ndarray((0, ))

    def add_action(self, action):  # Note that it could be a probabilistic action too.
        self.actions = np.append(self.actions, action)

    def __repr__(self):
        returned_string = ''
        if self.actions.size == 0:
            returned_string += f'There are no actions from state {self.name}.\n'
        else:
            returned_string += f'Actions from state {self.name}:\n'
            for action in self.actions:
                returned_string += repr(action)
            returned_string += '\n'

        return returned_string
