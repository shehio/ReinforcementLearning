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
            returned_string += f'State {self.name} has value of {self.value} and no actions.\n\n'
        else:
            returned_string += f'State {self.name} has value of {self.value} and actions:\n'
            for counter, action in enumerate(self.actions):
                returned_string += f'{str(counter + 1)}. {repr(action)}'
            returned_string += '\n\n'

        return returned_string
